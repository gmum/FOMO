#!/bin/bash
# =============================================================================
# Generate the ceiling videos for the near2 mapping targets.
#
#   sbatch --array=0-29 slurm/gen_ceiling_gh200.sbatch
#
# Index = one (variant, target) cell, twenty videos each. Ten targets times
# three variants is thirty cells, six hundred videos.
#
#   A  the v7 prompt with the noun swapped and nothing else
#   B  the same scene with the contradicted attributes repaired
#   C  one plain prompt on twenty seeds
#
# List the cells with:
#   python3 benchmark/generate_ceiling.py --list
#
# Output: videos/_ceiling/<variant>/<target slug>/<NN>.mp4
# =============================================================================

#SBATCH --job-name=ceil_gen
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/ceil_gen_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/ceil_gen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
ENV_DIR=$BASE_DIR/conda/envs/eraser

export HF_HOME=$REPO_DIR/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$REPO_DIR/.cache/torch
export TMPDIR=$REPO_DIR/.cache/tmp
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$REPO_DIR/logs" "$TMPDIR"

module purge
module load CUDA/12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
exec python3 benchmark/generate_ceiling.py --cell "$IDX" --out_root "${OUT_ROOT:-videos/_ceiling}"
