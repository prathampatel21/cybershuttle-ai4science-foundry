#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
workflow=${WORKFLOW_ROOT:-../../docs/tutorials/chgnet-pratham}
workflow=$(cd "$workflow" && pwd)
source "$workflow/.venv/bin/activate"
id=${EXPERIMENT_ID:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}}
mkdir -p runs
test ! -e "runs/$id" || { echo 'Output exists; choose another EXPERIMENT_ID.'; exit 1; }
exec > >(tee "runs/$id.execution.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "runs/$id.exit-status.txt"' EXIT
date -u
hostname
git rev-parse HEAD
python --version
python -m pip check
python -m unittest discover -s tests -v
python scripts/experiment.py --workflow-root "$workflow" --output "runs/$id" --device "${DEVICE:-cuda}"
python scripts/analyze.py "runs/$id"
python -m pip freeze > "runs/$id/environment-lock.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi > "runs/$id/nvidia-smi.txt"; fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol show job "$SLURM_JOB_ID" > "runs/$id/slurm-job.txt"; fi
