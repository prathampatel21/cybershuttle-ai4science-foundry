# CHGNet volume response proof of concept

Research question: how much does internal atomic relaxation change the predicted
energy-volume curve and effective isotropic stiffness of diamond silicon and
LiMnO2, and are the energy derivatives consistent with the predicted stress?

This extends the [Foundry inference contribution](../../docs/tutorials/chgnet-pratham/README.md)
from single-structure validation to a controlled mechanical-response experiment.
It uses checkpoint 0.3.0 and the same pinned CHGNet source and dependency lock.
No new DFT data or training is involved. The experiment does not establish
accuracy on held-out structures or reproduce a paper benchmark.

## Design

- Two eight-atom cells: ideal diamond Si generated with ASE at an initial lattice
  parameter of 5.43 Angstrom, and the official upstream LiMnO2 example.
- Seventeen volume factors from 0.88 to 1.12, inclusive, at spacing 0.015.
  The corresponding linear scale is the cube root of the volume factor.
- At each volume, evaluate frozen fractional sites, then optimize only atomic
  positions with ASE FIRE (maximum 200 steps, max force 0.03 eV/Angstrom).
  Each point starts independently from the reference fractional coordinates.
- Tighten five points around each sampled relaxed minimum to 0.01 eV/Angstrom.
- Compare pressure from stress, -trace(stress)/3, with -dE_total/dV using
  central volume differences of 0.1% and 0.5% at three scan volumes per branch.
  This is an internal numerical consistency check, not independent validation.
- Fit local quadratic curves with 5, 9 and 13 points centered on the sampled
  minimum. Report V0 and B_path = V0 d²E/dV² using energy and volume per atom;
  multiply eV/Angstrom³ by 160.21766208 for GPa. Edge minima and nonpositive
  curvature are flagged rather than extrapolated. Fit-window variation and
  force convergence are part of the result.

All cell vectors retain their shape. For anisotropic LiMnO2, B_path is a
constrained effective isotropic stiffness, not a fully hydrostatic modulus
obtained by relaxing cell shape. Scalar pressure may leave deviatoric stress.

## Reproduce on ICE

Use an allocated GPU compute session: one H100 or H200, four CPUs, 16 GB RAM,
one hour for setup and experiment. Confirm your course account/partition.
Clone into your own scratch directory; do not run compute on the login node.

```bash
mkdir -p "$HOME/scratch"
git clone --branch assignment5-chgnet-eos --single-branch \
  https://github.com/prathampatel21/cybershuttle-ai4science-foundry.git \
  "$HOME/scratch/chgnet-eos"
cd "$HOME/scratch/chgnet-eos/docs/tutorials/chgnet-pratham"
module load anaconda3/2023.03  # or provide Python 3.10 explicitly
set -o pipefail
mkdir -p runs
TORCH_BACKEND=cu121 DEPENDENCY_LOCK=evidence/6199752/dependencies-lock.txt \
  bash scripts/setup.sh 2>&1 | tee runs/setup.log
DEVICE=cuda bash scripts/run.sh  # established reference/symmetry checks
cd ../../../experiments/chgnet-eos
DEVICE=cuda bash scripts/run.sh
```

The lock already includes ASE. If reusing a previously validated environment,
set WORKFLOW_ROOT to that tutorial directory before the experiment command;
record this deviation and its package lock. Setup refuses to overwrite .venv.
CPU is explicitly supported with DEVICE=cpu, but runtime will differ.

Outputs: runs/<id>/scan.json, scan.csv, metadata.json, summary.json,
derivative-checks.json, refinements.json, structures/, relaxation/ and package
lock. The full terminal log and shell exit status sit alongside the directory.
A zero exit status means execution finished; inspect convergence and fit flags
before drawing scientific conclusions. Scripts never overwrite run evidence.

## Sources

- [Pinned CHGNet energy/stress implementation](https://github.com/CederGroupHub/chgnet/blob/b9a6c15860237297a5a5169660e83b2e61a78e76/chgnet/model/model.py)
- [Pinned ASE interface](https://github.com/CederGroupHub/chgnet/blob/b9a6c15860237297a5a5169660e83b2e61a78e76/chgnet/model/dynamics.py)
- [CHGNet scientific applications](https://chgnet.lbl.gov/)

Limitations and measured findings will be documented with the execution evidence.
Part 2 (presentation recording) is outside this experiment's current scope.
