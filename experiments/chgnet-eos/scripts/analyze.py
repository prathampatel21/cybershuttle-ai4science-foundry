"""Analyze the experiment without requiring torch, CHGNet or ASE."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

GPA = 160.21766208


def fit_window(volume, energy, half_width):
    volume, energy = np.asarray(volume), np.asarray(energy)
    i = int(np.argmin(energy))
    if i < half_width or i + half_width >= len(energy):
        return {'valid': False, 'reason': 'Minimum not bracketed with a complete symmetric fit window.'}
    sl = slice(i - half_width, i + half_width + 1)
    center = volume[i]
    x = (volume[sl] - center) / center
    a, b, c = np.polyfit(x, energy[sl], 2)
    v0 = center * (1 - b / (2 * a)) if a != 0 else -1
    if a <= 0 or not volume[i - half_width] < v0 < volume[i + half_width]:
        return {'valid': False, 'reason': 'Nonpositive curvature or fitted minimum outside window.'}
    residual = energy[sl] - np.polyval([a, b, c], x)
    return {'valid': True, 'n_points': 2 * half_width + 1, 'volume_A3_atom': float(v0),
            'stiffness_GPa': float(v0 * 2 * a / center**2 * GPA),
            'energy_rmse_meV_atom': float(np.sqrt(np.mean(residual**2)) * 1000),
            'coefficients': [float(a), float(b), float(c)], 'center_volume_A3_atom': float(center)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    run = args.run
    rows = json.loads((run / 'scan.json').read_text())
    checks = json.loads((run / 'derivative-checks.json').read_text())
    tight = json.loads((run / 'refinements.json').read_text())
    result = {'fits': [], 'checks': [], 'relaxation': []}
    for name in sorted({r['material'] for r in rows}):
        frozen = [r for r in rows if r['material'] == name and r['mode'] == 'frozen']
        relaxed = [r for r in rows if r['material'] == name and r['mode'] == 'relaxed']
        gains = [1000 * (a['energy_eV_atom'] - b['energy_eV_atom']) for a, b in zip(frozen, relaxed)]
        result['relaxation'].append({'material': name, 'converged': sum(r['converged'] for r in relaxed),
                                    'total': len(relaxed), 'min_gain_meV_atom': min(gains),
                                    'max_gain_meV_atom': max(gains),
                                    'max_steps': max(r['steps'] for r in relaxed)})
        for mode, branch in [('frozen', frozen), ('relaxed', relaxed)]:
            branch = sorted(branch, key=lambda r: r['index'])
            for width in (2, 4, 6):
                fit = fit_window([r['volume_A3_atom'] for r in branch], [r['energy_eV_atom'] for r in branch], width)
                fit.update(material=name, mode=mode, half_width=width)
                if mode == 'relaxed' and not all(r['converged'] for r in branch):
                    fit['relaxation_warning'] = 'Some scan points did not reach the force criterion.'
                result['fits'].append(fit)
            for h in (.001, .005):
                errors = [r['abs_error_GPa'] for r in checks if r['material'] == name and r['mode'] == mode and r['relative_volume_step'] == h]
                result['checks'].append({'material': name, 'mode': mode, 'relative_volume_step': h,
                                         'max_abs_error_GPa': max(errors), 'count': len(errors)})
        selected = [r for r in tight if r['material'] == name]
        base = {r['index']: r for r in relaxed}
        result.setdefault('refinement', []).append({'material': name, 'converged': sum(r['converged'] for r in selected),
            'total': len(selected), 'max_extra_energy_drop_meV_atom': max((base[r['index']]['energy_eV_atom'] - r['energy_eV_atom']) * 1000 for r in selected),
            'tight_fit': fit_window([r['volume_A3_atom'] for r in selected], [r['energy_eV_atom'] for r in selected], 2)})
    (run / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    keys = ['material', 'mode', 'index', 'volume_factor', 'volume_A3_atom', 'energy_eV_atom', 'pressure_GPa', 'max_force_eV_A', 'converged', 'steps']
    with (run / 'scan.csv').open('w') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
