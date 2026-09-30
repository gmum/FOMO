#!/bin/bash
# =============================================================================
# Generate videos from a hand-written prompt list.
#
# The list is configs/adhoc_prompts.txt, one prompt per line. The array splits
# it into slices of CHUNK lines; array index N renders lines N*CHUNK onward.
# Submit through slurm/run_adhoc.sh, which counts the lines and sizes the
# array.
#
# All slices write into the same folder and number their output by line, so
# NN.mp4 is always line NN of the file, whichever job produced it. The seed is
# seed + line, so every prompt gets its own noise and the same line rendered
# through the base model and through an adapter gets the same noise twice.
#
# Base model by default; LORA points at a checkpoint directory instead.
#
# Output: videos/adhoc/<TAG>/NNN.mp4
# =============================================================================

#SBATCH --job-name=adhoc_gen
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/adhoc_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/adhoc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
ENV_DIR=${ENV_DIR:-$BASE_DIR/conda/envs/eraser}

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

: "${ADHOC_FILE:=configs/adhoc_prompts.txt}"
: "${ADHOC_CHUNK:=10}"
: "${TAG:=base}"
: "${LORA:=}"
: "${LORA_WEIGHT:=1.0}"
: "${NUM_FRAMES:=17}"
: "${HEIGHT:=720}"
: "${WIDTH:=1280}"
: "${SEED:=42}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( IDX * ADHOC_CHUNK ))

echo "Host : $(hostname)  $(date)"
echo "List : $ADHOC_FILE"
echo "Slice: lines $START .. $(( START + ADHOC_CHUNK - 1 ))  (task $IDX)"
echo "Tag  : $TAG"
[ -n "$LORA" ] && echo "LoRA : $LORA (weight $LORA_WEIGHT)"
echo "----------------------------------------------------------"

CMD=(python3 benchmark/generate_adhoc.py
    --file "$ADHOC_FILE"
    --out "videos/adhoc/$TAG"
    --start "$START"
    --count "$ADHOC_CHUNK"
    --height "$HEIGHT" --width "$WIDTH"
    --num_frames "$NUM_FRAMES" --seed "$SEED")

if [ -n "$LORA" ]; then
    CMD+=(--lora "$LORA" --lora_weight "$LORA_WEIGHT")
fi

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
