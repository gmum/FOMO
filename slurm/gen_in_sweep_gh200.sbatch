#!/bin/bash
# =============================================================================
# Generate the table column for every variant of the Imagenette sweep.
#
#   sbatch --array=0-399%8 --export=ALL,LEARNING_RATE=1e-3 slurm/gen_in_sweep_gh200.sbatch
#   sbatch --array=320-329 ...                              # one variant
#
# Index = variant * 10 + class, where variant is the line of
# configs/in_variants.tsv and class is the Imagenette class being generated.
# A column needs all ten classes, otherwise PSR cannot be computed.
#
# Method label is imap_<mapping>_lr<lr>, so the four mappings sit side by side
# in one column directory and each forms its own table row.
#
# Variants without trained weights exit quietly, so the array can be submitted
# as a dependency and re-run later to fill gaps.
# =============================================================================

#SBATCH --job-name=in_gen
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/in_gen_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/in_gen_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
export REPO_DIR
cd "$REPO_DIR"

TABLE="${TABLE:-configs/in_variants.tsv}"
: "${LEARNING_RATE:=1e-3}"
: "${MASK_MODE:=unclamped}"
: "${IMAP_ALPHA:=7}"
: "${CHECKPOINT:=}"          # empty means the newest checkpoint
LR_TAG=$(echo "$LEARNING_RATE" | tr -d '.-')

IDX="${SLURM_ARRAY_TASK_ID:-0}"
VAR_IDX=$((IDX / 10))
CLS_IDX=$((IDX % 10))

LINE="$(sed -n "$((VAR_IDX + 1))p" "$TABLE")"
[ -n "$LINE" ] || { echo "Nothing at variant $VAR_IDX"; exit 0; }
IFS=$'\t' read -r CONCEPT MAPPING _ _ _ <<< "$LINE"

SLUG=$(CONCEPT="$CONCEPT" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["CONCEPT"]))')
WEIGHTS="$REPO_DIR/outputs/in_sweep/${SLUG}_${MAPPING}_lr${LR_TAG}/${MASK_MODE}_alpha${IMAP_ALPHA}"
[ -n "$CHECKPOINT" ] && WEIGHTS="$WEIGHTS/$CHECKPOINT"

echo "Task $IDX -> variant $VAR_IDX ($CONCEPT / $MAPPING), class index $CLS_IDX"
echo "  weights: $WEIGHTS"
if ! compgen -G "$WEIGHTS/checkpoint-*" > /dev/null 2>&1 \
   && [ ! -f "$WEIGHTS/pytorch_lora_weights.safetensors" ]; then
    echo "SKIP: no trained weights under $WEIGHTS yet."
    exit 0
fi

export CLASS_SET="${CLASS_SET:-imagenette}"
export PROMPT_SET="${PROMPT_SET:-v7}"
export WIDTH="${WIDTH:-1280}"
export HEIGHT="${HEIGHT:-720}"
export ERASED="$CONCEPT"
# The checkpoint goes into the method label whenever one is pinned.
# Without it, generating step 100 and then step 250 would land in the
# same directory and --skip_existing would silently keep the first.
METHOD_LABEL="imap_${MAPPING}_lr${LR_TAG}"
[ -n "$CHECKPOINT" ] && METHOD_LABEL="${METHOD_LABEL}_${CHECKPOINT#checkpoint-}"
export MODELS="$METHOD_LABEL=$WEIGHTS"
echo "  method : $METHOD_LABEL"
export NO_BASE=1
export SLURM_ARRAY_TASK_ID="$CLS_IDX"

exec bash benchmark/run_generate.sbatch
