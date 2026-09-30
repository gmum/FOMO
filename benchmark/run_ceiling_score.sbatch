#!/bin/bash
# =============================================================================
# Normalise CLIP-tgt between the floor and the three ceilings.
#
#   sbatch --export=ALL,CLIP_CSV=results/imagenette_v7_1280x720/clip_score_v2.csv,\
# METHOD=imap_near2_lr1e3 benchmark/run_ceiling_score.sbatch
#
# Minutes on one card: the floor and the method's own score are read from the
# CSV, and only the six hundred ceiling videos are encoded.
#
# Output: results/<tag>/ceiling_score.csv beside the CSV it was given.
# =============================================================================

#SBATCH --job-name=ceil_score
#SBATCH --output=logs/ceil_score_%j.out
#SBATCH --error=logs/ceil_score_%j.err
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

: "${CLIP_CSV:?set CLIP_CSV to the clip_score CSV holding the floor}"
: "${METHOD:=imap_near2_lr1e3}"
: "${CEILING_ROOT:=videos/_ceiling}"
: "${VARIANTS:=A B C}"
: "${CLIP_MODEL:=openai/clip-vit-large-patch14}"
: "${OUT:=}"

# shellcheck disable=SC2206
VARIANT_ARR=($VARIANTS)

CMD=(python3 benchmark/ceiling_score.py
    --clip_csv "$CLIP_CSV"
    --method "$METHOD"
    --ceiling_root "$CEILING_ROOT"
    --variants "${VARIANT_ARR[@]}"
    --clip_model "$CLIP_MODEL")
if [ -n "$OUT" ]; then CMD+=(--out "$OUT"); fi

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
echo "----------------------------------------------------------"
exec "${CMD[@]}"
