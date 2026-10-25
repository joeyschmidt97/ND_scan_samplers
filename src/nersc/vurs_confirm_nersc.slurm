#!/bin/bash
#SBATCH --job-name=vurs-confirm
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=slurm-vurs-confirm-%j.out
#
# VURS confirmation trajectories on one Perlmutter CPU node, one worker per core.
# Frozen manifests: manifests/vurs-confirm-5d.json (5732ecf), -6d.json (9215f5a).
# Finished trajectories are never recomputed, so a resubmit resumes; copy a
# partial local outputs/vurs-confirm-<d>d-n512/confirm here first to skip them.
#
#   DIM=5 MANIFEST=5732ecfe243f662cf406cb56eb3457d8d959c71a sbatch --account m2116 src/nersc/vurs_confirm_nersc.slurm
#   DIM=6 MANIFEST=9215f5a5b52f0714e9b5f22a5d15c1060c1fc46f sbatch --account m2116 src/nersc/vurs_confirm_nersc.slurm

set -euo pipefail
: "${PSCRATCH:?PSCRATCH is not set}"
: "${DIM:?set DIM=5 or DIM=6}"
: "${MANIFEST:?set MANIFEST to the full manifest commit hash}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_PREFIX="${UQ_ENV:-$PSCRATCH/conda-envs/uq-discontinuities}"
export SG_LIB_PATH="${SG_LIB_PATH:-$PSCRATCH/src/sensitivity-driven-sparse-grid-approx}"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO_ROOT"
export PYTHONPATH="$(dirname "$REPO_ROOT")${PYTHONPATH:+:$PYTHONPATH}"   # imports are rooted at the parent of the repo

OUT="outputs/vurs-confirm-${DIM}d-n512"
mkdir -p "$OUT"
python -m ND_scan_samplers.scripts.experiments.vurs_confirm --namespace confirm --manifest-commit "$MANIFEST" \
  --dim "$DIM" --budget 512 --blocks 0-159 --output "$OUT" --workers 128
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_score --input "$OUT/confirm" --manifest-commit "$MANIFEST" \
  --workers 128
