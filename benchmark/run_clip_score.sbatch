#!/bin/bash
# =============================================================================
# CLIP similarity for the erased class of every method in a sweep.
#
#   sbatch benchmark/run_clip_score.sbatch
#   sbatch --export=ALL,METHODS="imap_near_lr1e3 imap_far_lr1e3" \
#          benchmark/run_clip_score.sbatch
#
# One job covers every method and both CLIP models; the work is a few tens of
# thousands of image encodings, which is minutes on one card.
#
# Output: results/<tag>/clip_score.csv
# =============================================================================

#SBATCH --job-name=clip_score
#SBATCH --output=logs/clip_score_%j.out
#SBATCH --error=logs/clip_score_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.cache/tmp}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs "$TMPDIR"

source benchmark/env.sh
require_modules torch transformers

: "${METHODS:=imap_null_lr1e3 imap_bg_lr1e3 imap_near_lr1e3 imap_far_lr1e3}"
: "${TABLE:=configs/in_variants.tsv}"
: "${CLASS_SET:=imagenette}"
: "${PROMPT_SET:=v7}"
: "${WIDTH:=1280}"
: "${HEIGHT:=720}"
: "${CLIP_MODELS:=openai/clip-vit-large-patch14 openai/clip-vit-base-patch32}"
# Empty means the default path under results/<tag>. Set it whenever a second
# sweep is scored, otherwise the first sweep's file is silently overwritten.
: "${OUT:=}"

# shellcheck disable=SC2206
METHOD_ARR=($METHODS)
# shellcheck disable=SC2206
MODEL_ARR=($CLIP_MODELS)

CMD=(python3 benchmark/clip_score.py
    --methods "${METHOD_ARR[@]}"
    --clip_models "${MODEL_ARR[@]}"
    --table "$TABLE"
    --class_set "$CLASS_SET"
    --prompt_set "$PROMPT_SET"
    --width "$WIDTH" --height "$HEIGHT")

if [ -n "$OUT" ]; then CMD+=(--out "$OUT"); fi

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
echo "----------------------------------------------------------"
exec "${CMD[@]}"
