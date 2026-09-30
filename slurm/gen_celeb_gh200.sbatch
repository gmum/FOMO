#!/bin/bash
# =============================================================================
# Generate the identity columns, baseline or unlearned.
#
#   MODE=base:
#     sbatch --array=0-7 --export=ALL,MODE=base slurm/gen_celeb_gh200.sbatch
#     Index = person. Eight tasks, the floor every column is compared against.
#
#   MODE=column (default):
#     sbatch --array=0-79 slurm/gen_celeb_gh200.sbatch
#     Index = variant * 8 + person, where variant is the line of
#     configs/celeb_variants.tsv. Every column generates all eight people, so
#     the damage to the seven identities that were not erased is measurable
#     the same way PSR measures it for objects.
#
# Variants without trained weights exit quietly, so the array can be submitted
# as a dependency and re-run later to fill gaps.
#
# Output: videos/celeb/_shared/base/<person>/base/<NN>.mp4
#         videos/celeb/unlearn_<erased>/<person>/imap_<mapping>_lr<lr>/<NN>.mp4
# =============================================================================

#SBATCH --job-name=celeb_gen
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/celeb_gen_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/celeb_gen_%A_%a.err
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

MODE="${MODE:-column}"
TABLE="${TABLE:-configs/celeb_variants.tsv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos/celeb}"
SWEEP_DIR="${SWEEP_DIR:-celeb_sweep}"
PROMPTS_MODULE="${PROMPTS_MODULE:-celeb_prompts}"
: "${LEARNING_RATE:=1e-3}"
: "${MASK_MODE:=unclamped}"
: "${IMAP_ALPHA:=7}"
: "${CHECKPOINT:=}"
# Use only the first N prompts. 0 keeps all of them.
: "${NUM_PROMPTS:=0}"
# Appended to the method label. IMAP_ALPHA changes the weights directory but
# not the label, so two alpha settings would otherwise overwrite each other's
# videos -- pass METHOD_SUFFIX=_a3 to keep them apart.
: "${METHOD_SUFFIX:=}"
# Scaling the adapter at inference walks along the erasure/damage trade-off
# without retraining anything: the same weights applied at half strength are a
# different operating point, not a different method.
: "${LORA_WEIGHT:=1.0}"
LR_TAG=$(echo "$LEARNING_RATE" | tr -d '.-')

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# The person index runs over the alphabetical list in celeb_prompts, which is
# also the order the column directories are built in.
N_PEOPLE=$(M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(len(importlib.import_module(os.environ["M"]).PEOPLE))')

if [ "$MODE" = "base" ]; then
    PERSON=$(IDX="$IDX" M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(importlib.import_module(os.environ["M"]).PEOPLE[int(os.environ["IDX"])])')
    SLUG=$(PERSON="$PERSON" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["PERSON"]))')
    echo "Task $IDX | base | $PERSON"
    exec python3 benchmark/generate_celeb.py \
        --person "$PERSON" \
        --output_dir "$VIDEOS_ROOT/_shared/base/$SLUG" \
        --method base \
        --num_prompts "$NUM_PROMPTS" \
        --prompts_module "$PROMPTS_MODULE"
fi

VAR_IDX=$((IDX / N_PEOPLE))
PER_IDX=$((IDX % N_PEOPLE))

LINE="$(sed -n "$((VAR_IDX + 1))p" "$TABLE")"
[ -n "$LINE" ] || { echo "Nothing at variant $VAR_IDX"; exit 0; }
IFS=$'\t' read -r ERASED MAPPING _ _ _ <<< "$LINE"

PERSON=$(IDX="$PER_IDX" M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(importlib.import_module(os.environ["M"]).PEOPLE[int(os.environ["IDX"])])')
ERASED_SLUG=$(P="$ERASED" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["P"]))')
PERSON_SLUG=$(P="$PERSON" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["P"]))')

WEIGHTS="$REPO_DIR/outputs/${SWEEP_DIR}/${ERASED_SLUG}_${MAPPING}_lr${LR_TAG}/${MASK_MODE}_alpha${IMAP_ALPHA}"
[ -n "$CHECKPOINT" ] && WEIGHTS="$WEIGHTS/$CHECKPOINT"

echo "Task $IDX -> variant $VAR_IDX (erase '$ERASED' / $MAPPING), generating '$PERSON'"
echo "  weights: $WEIGHTS"
if ! compgen -G "$WEIGHTS/checkpoint-*" > /dev/null 2>&1 \
   && [ ! -f "$WEIGHTS/pytorch_lora_weights.safetensors" ]; then
    # Quiet by default, so the array can be submitted as a dependency and
    # re-run later to fill gaps. STRICT=1 when the weights are supposed to be
    # there already: a column that silently generates nothing looks exactly
    # like a column that erased everything.
    if [ "${STRICT:-0}" = "1" ]; then
        echo "ERROR: no trained weights under $WEIGHTS" >&2
        exit 1
    fi
    echo "SKIP: no trained weights under $WEIGHTS yet."
    exit 0
fi

METHOD_LABEL="imap_${MAPPING}_lr${LR_TAG}${METHOD_SUFFIX}"
[ -n "$CHECKPOINT" ] && METHOD_LABEL="${METHOD_LABEL}_${CHECKPOINT#checkpoint-}"
echo "  method : $METHOD_LABEL"

exec python3 benchmark/generate_celeb.py \
    --person "$PERSON" \
    --output_dir "$VIDEOS_ROOT/unlearn_${ERASED_SLUG}/${PERSON_SLUG}" \
    --method "$METHOD_LABEL" \
    --lora "$WEIGHTS" \
    --lora_weight "$LORA_WEIGHT" \
    --num_prompts "$NUM_PROMPTS" \
    --prompts_module "$PROMPTS_MODULE"
