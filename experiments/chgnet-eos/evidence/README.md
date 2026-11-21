# Run 7229001 evidence

Executed November 21, 2026 on ICE Tesla V100-PCIE-16GB; one GPU, four CPUs,
16 GB requested RAM, account coc, partition ice-gpu. The allocation was obtained
with the following command from scratch storage:

```bash
cd "$HOME/scratch"
srun --account=coc --partition=ice-gpu --gres=gpu:1 --cpus-per-task=4 \
  --mem=16G --time=01:00:00 --pty bash
```

The fresh experiment clone used branch assignment5-chgnet-eos at
`e26db6602f906a500aebca6276cbf23e232911ed`. The original CHGNet environment
and pinned upstream source remained at WORKFLOW_ROOT. The actual commands were:

```bash
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$HOME/scratch/chgnet-eos-mpl"
cd "$HOME/scratch/chgnet-eos/experiments/chgnet-eos"
WORKFLOW_ROOT="$HOME/chgnet-inference" DEVICE=cuda bash scripts/run.sh
python "$HOME/chgnet-inference/scripts/infer.py" --device cuda \
  --output runs/7229001/reference-validation.json \
  > runs/7229001/reference-validation.log 2>&1
```

The home quota was full, preventing OnDemand session staging. No files were
deleted: direct Slurm allocation and scratch outputs resolved the execution
blocker. The reused environment passed pip check, two numerical-analysis tests,
and fresh original numerical/symmetry checks. Script and input hashes are in
metadata.json. The setup was reused, not rebuilt during this run; the standard
fresh-setup instructions and dependency lock remain in the tutorial.

`7229001.execution.log` and `7229001.exit-status.txt` are experiment-level files.
The run directory contains the raw scan, metadata, CSV and summary, all FIRE
logs, initial and relaxed CIF files, force-threshold refinements, derivative
checks, environment lock, GPU/scheduler records, and reference validation.
The scheduler snapshot was taken while running. Exiting the allocated shell
after collecting evidence released the GPU; squeue subsequently showed no job.

The PDF-generation script later computed pressure-slope-analysis.json from the
same stored predictions. This is supplementary local postprocessing, not a
second inference experiment. The raw inference and experiment script remain
unchanged from the recorded execution revision.
