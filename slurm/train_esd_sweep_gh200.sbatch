#!/bin/bash
# =============================================================================
# ESD sweep over the ten Imagenette classes, twice: with and without L_V.
#
# The point of the pair is not the ESD row on its own. It is that L_V is a
# term, not a method -- if it lifts preservation when bolted onto somebody
# else's objective, that is a far stronger claim than showing it helps inside
# our own.
#
#   Index = class * 2 + variant
#     variant 0  ->  esd     (lamb_v = 0)     plain baseline
#     variant 1  ->  esd_v   (lamb_v = LAMB_V) baseline + our term
#
#   sbatch slurm/train_esd_sweep_gh200.sbatch                     all twenty
#   sbatch --array=0,1 slurm/train_esd_sweep_gh200.sbatch         tench only
#   sbatch --array=0-19:2 slurm/train_esd_sweep_gh200.sbatch      plain ESD only
#   sbatch --export=ALL,LEARNING_RATE=5e-5 ...                    lr sweep
#
# Weights: outputs/<SWEEP_DIR>/<class>_<variant>_lr<lr>/checkpoint-*
# =============================================================================

#SBATCH --job-name=esd_train
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_train_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_train_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --array=0-19

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
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/outputs" "$TMPDIR"

module purge
module load CUDA/12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

: "${CONFIG:=configs/esd_imagenette.yaml}"
: "${CLASS_SET:=imagenette}"
# A plain text file, one concept per line, blank lines and # comments ignored.
# Set it to erase anything that is not an Imagenette class; leave it empty to
# sweep the benchmark. Array size is twice the number of concepts.
: "${CONCEPTS_FILE:=}"
: "${SWEEP_DIR:=esd_sweep}"
: "${LEARNING_RATE:=2e-5}"
: "${LAMB_V:=0.5}"
: "${V_STEPS:=1}"
: "${V_BLOCK_MIN:=20}"
: "${V_EARLY_PROB:=0.5}"
: "${MAX_TRAIN_STEPS:=250}"
: "${CHECKPOINTING_STEPS:=10}"
: "${RANK:=8}"
: "${LORA_ALPHA:=8}"
: "${TARGET_MODULES:=attn.add_q_proj,attn.add_k_proj}"
# Training resolution, not generation resolution. See the note in the config:
# L_V runs without gradient checkpointing and 1280x720x17 does not fit in 95 GiB.
: "${HEIGHT:=288}"
: "${WIDTH:=512}"
: "${TRAIN_NUM_FRAMES:=17}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CLS_IDX=$((IDX / 2))
VAR_IDX=$((IDX % 2))

CONCEPT=$(I="$CLS_IDX" S="$CLASS_SET" F="$CONCEPTS_FILE" python3 -c '
import os, sys
sys.path.insert(0, "benchmark")
path = os.environ["F"]
if path:
    with open(path, encoding="utf-8") as handle:
        classes = [line.strip() for line in handle
                   if line.strip() and not line.lstrip().startswith("#")]
else:
    from common import CLASS_SETS
    classes = CLASS_SETS[os.environ["S"]]
i = int(os.environ["I"])
if i >= len(classes):
    sys.exit("ERROR: concept index %d out of range 0..%d" % (i, len(classes) - 1))
print(classes[i])
')
SLUG=$(CONCEPT="$CONCEPT" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["CONCEPT"]))')

if [ "$VAR_IDX" = "0" ]; then
    VARIANT=esd
    USE_LAMB_V=0.0
else
    VARIANT=esd_v
    USE_LAMB_V="$LAMB_V"
fi

LR_TAG=$(echo "$LEARNING_RATE" | tr -d '.-')
OUTPUT_DIR="$REPO_DIR/outputs/${SWEEP_DIR}/${SLUG}_${VARIANT}_lr${LR_TAG}"
DATASET_DIR="$REPO_DIR/data/esd/${SLUG}"
mkdir -p "$OUTPUT_DIR" "$DATASET_DIR"

echo "Host: $(hostname)"
echo "Task $IDX | class $CLS_IDX '$CONCEPT' | variant $VARIANT"
echo "  lamb_v : $USE_LAMB_V   lr $LEARNING_RATE   steps $MAX_TRAIN_STEPS"
echo "  output : $OUTPUT_DIR"

# --use_empty_neutral_prompt makes the neutral side the empty prompt, which is
# what ESD erases toward. It is the same choice the null mapping makes, so the
# two are erasing in the same direction and differ only in how.
CMD=(
    python train_hunyuan_erase_lora.py
    --config "$CONFIG"
    --concept "$CONCEPT"
    --output_dir "$OUTPUT_DIR"
    --dataset_dir "$DATASET_DIR"
    --use_empty_neutral_prompt
    --prompt_only_esd
    --bootstrap_dataset_if_missing
    --gradient_checkpointing
    --learning_rate "$LEARNING_RATE"
    --rank "$RANK" --lora_alpha "$LORA_ALPHA"
    --target_modules "$TARGET_MODULES"
    --max_train_steps "$MAX_TRAIN_STEPS"
    --checkpointing_steps "$CHECKPOINTING_STEPS"
    --train_width "$WIDTH" --train_height "$HEIGHT"
    --train_num_frames "$TRAIN_NUM_FRAMES"
    --lamb_v "$USE_LAMB_V"
    --v_steps "$V_STEPS"
    --v_block_min "$V_BLOCK_MIN"
    --v_early_prob "$V_EARLY_PROB"
)

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
echo "----------------------------------------------------------"
exec "${CMD[@]}"
