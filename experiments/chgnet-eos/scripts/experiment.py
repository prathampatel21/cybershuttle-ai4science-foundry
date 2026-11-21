"""Fixed-shape volume scans: frozen sites versus CHGNet/ASE relaxation.

No DFT or experimental accuracy claim is made. Each volume starts independently
from the same reference fractional coordinates, avoiding continuation bias.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.optimize import FIRE
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

GPA = 160.21766208  # Matches the pinned CHGNet stress conversion.


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def observe(atoms):
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False) * GPA
    return {
        'energy_eV_atom': float(atoms.get_potential_energy() / len(atoms)),
        'volume_A3_atom': float(atoms.get_volume() / len(atoms)),
        'max_force_eV_A': float(np.linalg.norm(forces, axis=1).max()),
        'pressure_GPa': float(-np.trace(stress) / 3),
        'stress_GPa': stress.tolist(),
    }


def finite_difference(atoms, calculator, h):
    # Fixed fractional coordinates, including when checking relaxed configurations.
    energies = []
    for scale in (1 - h, 1 + h):
        perturbed = atoms.copy()
        perturbed.set_cell(atoms.cell * scale ** (1 / 3), scale_atoms=True)
        perturbed.calc = calculator
        energies.append(float(perturbed.get_potential_energy()))
    return -(energies[1] - energies[0]) / (2 * h * atoms.get_volume()) * GPA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workflow-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--fmax', type=float, default=0.03)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Choose a new output directory; previous evidence is never overwritten.')
    out.mkdir(parents=True)
    (out / 'structures').mkdir()
    (out / 'relaxation').mkdir()
    upstream = args.workflow_root.resolve() / 'artifacts/upstream'
    cfg = json.loads((args.workflow_root / 'config/workflow.json').read_text())
    sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != cfg['source_commit']:
        raise SystemExit('Upstream revision does not match the established workflow.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available; allocate a GPU first.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    model = CHGNet.load(model_name='0.3.0', use_device=args.device)
    model.eval()
    calc = CHGNetCalculator(model=model, use_device=args.device)
    silicon = bulk('Si', 'diamond', a=5.43, cubic=True)
    limno2 = AseAtomsAdaptor.get_atoms(Structure.from_file(upstream / cfg['input_relative_path']))
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': sha, 'checkpoint': '0.3.0',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256((upstream / cfg['input_relative_path']).read_bytes()).hexdigest(),
        'packages': {p: md.version(p) for p in ['chgnet', 'torch', 'numpy', 'ase', 'pymatgen']},
        'python': platform.python_version(), 'hostname': platform.node(),
        'device': args.device,
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'seed': 42, 'threads': 4, 'fmax_eV_A': args.fmax, 'max_steps': args.steps,
        'volume_factors': np.linspace(.88, 1.12, 17).tolist(),
        'silicon_initial_lattice_A': 5.43,
        'constraint': 'Fixed cell shape at each isotropically scaled volume; only sites relaxed.',
        'slurm': {k: os.environ.get(k) for k in ['SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_JOB_ACCOUNT']},
    }
    write_json(out / 'metadata.json', metadata)
    records, derivative_checks, refinements = [], [], []
    start = time.perf_counter()
    for name, reference in [('Si', silicon), ('LiMnO2', limno2)]:
        AseAtomsAdaptor.get_structure(reference).to(filename=str(out / 'structures' / f'{name}_initial.cif'))
        saved = []
        for index, factor in enumerate(metadata['volume_factors']):
            atoms = reference.copy()
            atoms.set_cell(reference.cell * factor ** (1 / 3), scale_atoms=True)
            atoms.calc = calc
            frozen = {'material': name, 'mode': 'frozen', 'index': index, 'volume_factor': factor, **observe(atoms)}
            records.append(frozen)
            opt = FIRE(atoms, logfile=str(out / 'relaxation' / f'{name}_{index:02}.log'))
            converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
            relaxed = {'material': name, 'mode': 'relaxed', 'index': index, 'volume_factor': factor,
                       'converged': converged, 'steps': opt.nsteps, **observe(atoms)}
            records.append(relaxed)
            AseAtomsAdaptor.get_structure(atoms).to(filename=str(out / 'structures' / f'{name}_{index:02}.cif'))
            saved.append(atoms.copy())
            print(json.dumps(relaxed), flush=True)
            if index in (4, 8, 12):
                for mode, config in [('frozen', reference.copy()), ('relaxed', atoms.copy())]:
                    if mode == 'frozen':
                        config.set_cell(reference.cell * factor ** (1 / 3), scale_atoms=True)
                    config.calc = calc
                    pressure = observe(config)['pressure_GPa']
                    for h in (.001, .005):
                        fd = finite_difference(config, calc, h)
                        derivative_checks.append({'material': name, 'mode': mode, 'index': index,
                                                  'relative_volume_step': h, 'model_pressure_GPa': pressure,
                                                  'finite_difference_GPa': fd, 'abs_error_GPa': abs(fd - pressure)})
            write_json(out / 'scan.json', records)
            write_json(out / 'derivative-checks.json', derivative_checks)
        # Tighten force convergence around the sampled relaxed minimum.
        local = [r for r in records if r['material'] == name and r['mode'] == 'relaxed']
        minimum = min(local, key=lambda r: r['energy_eV_atom'])['index']
        for i in range(max(0, minimum - 2), min(17, minimum + 3)):
            atoms = saved[i]
            atoms.calc = calc
            opt = FIRE(atoms, logfile=str(out / 'relaxation' / f'{name}_{i:02}_tight.log'))
            ok = bool(opt.run(fmax=.01, steps=args.steps))
            refinements.append({'material': name, 'index': i, 'fmax_eV_A': .01, 'converged': ok,
                                'steps': opt.nsteps, **observe(atoms)})
        write_json(out / 'refinements.json', refinements)
    metadata['elapsed_seconds'] = time.perf_counter() - start
    if args.device == 'cuda':
        metadata['gpu_peak_allocated_MiB'] = torch.cuda.max_memory_allocated() / 1024**2
    write_json(out / 'metadata.json', metadata)
    print('EXPERIMENT COMPLETED; assess convergence and fits separately from execution.', flush=True)


if __name__ == '__main__':
    main()
