#!/bin/bash
# =============================================================================
# IMAP-H LoRA sweep over configs/in_variants.tsv: ten Imagenette classes
# times four kinds of mapping. TABLE points it at any other table with the
# same five columns.
#
# Index = line of the table, whose columns are
#   concept <TAB> mapping <TAB> target_concept <TAB> source_prompt <TAB> target_prompt
#
# NONE means "empty": a genuinely empty column would be swallowed, because tab
# counts as IFS whitespace and consecutive tabs collapse.
#
#   sbatch --export=ALL,LEARNING_RATE=1e-3 slurm/train_in_sweep_gh200.sbatch
#   sbatch --array=32-35 --export=ALL,LEARNING_RATE=1e-3 ...        one class
#   sbatch --array=0 --export=ALL,TABLE=configs/concept_variants.tsv,\
#          SWEEP_DIR=concept_sweep,TAG_SUFFIX=_clamp,MASK_MODE=clamped ...
#
# Weights: outputs/<SWEEP_DIR>/<class>_<mapping>_lr<lr><TAG_SUFFIX>/
#          <mask_mode>_alpha<alpha>/checkpoint-*
# =============================================================================

#SBATCH --job-name=in_train
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/in_train_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/in_train_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --array=0-39

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

TABLE="${TABLE:-configs/in_variants.tsv}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((IDX + 1))p" "$TABLE")"
[ -n "$LINE" ] || { echo "ERROR: no line $((IDX + 1)) in $TABLE" >&2; exit 2; }
IFS=$'\t' read -r CONCEPT MAPPING TARGET_CONCEPT SOURCE_PROMPT TARGET_PROMPT <<< "$LINE"
if [ "$TARGET_CONCEPT" = "NONE" ]; then TARGET_CONCEPT=""; fi
if [ "$TARGET_PROMPT" = "NONE" ];  then TARGET_PROMPT="";  fi

: "${SWEEP_DIR:=in_sweep}"
: "${TAG_SUFFIX:=}"
: "${LEARNING_RATE:=1e-3}"
: "${IMAP_ALPHA:=7}"
: "${MASK_MODE:=unclamped}"
: "${TARGET_MODULES:=attn.add_q_proj,attn.add_k_proj}"
: "${RANK:=8}"
: "${LORA_ALPHA:=8}"
: "${IMAP_MODE:=object}"
: "${LAMB_V:=0.5}"
: "${V_STEPS:=1}"
: "${V_BLOCK_MIN:=20}"
: "${V_EARLY_PROB:=0.5}"
: "${V_MASK_MODE:=background}"
: "${MAX_TRAIN_STEPS:=250}"
: "${CHECKPOINTING_STEPS:=10}"
: "${EVAL_STEPS:=125}"
: "${SKIP_EVAL:=1}"
: "${TRAIN_NUM_FRAMES:=17}"
: "${HEIGHT:=720}"
: "${WIDTH:=1280}"
: "${EVAL_HEIGHT:=720}"
: "${EVAL_WIDTH:=1280}"
: "${EVAL_NUM_FRAMES:=17}"
: "${EVAL_INFERENCE_STEPS:=30}"
: "${GUIDANCE_SCALE:=6.0}"
: "${MIN_TARGET_STEP:=0}"
: "${MAX_TARGET_STEP:=29}"
: "${SEED:=42}"
: "${EVAL_SEED:=42}"

case "$MASK_MODE" in
    clamped|unclamped) ;;
    *) echo "ERROR: MASK_MODE must be clamped or unclamped, got '$MASK_MODE'" >&2; exit 2 ;;
esac

LR_TAG=$(echo "$LEARNING_RATE" | tr -d '.-')
SLUG=$(CONCEPT="$CONCEPT" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["CONCEPT"]))')
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/${SWEEP_DIR}/${SLUG}_${MAPPING}_lr${LR_TAG}${TAG_SUFFIX}}"
OUTPUT_DIR=$OUTPUT_ROOT/${MASK_MODE}_alpha${IMAP_ALPHA}
mkdir -p "$OUTPUT_DIR"

PAIRS="$OUTPUT_DIR/pairs.jsonl"
EVALS="$OUTPUT_DIR/eval_prompts.jsonl"

# Concepts travel in the pair file, because the trainer reads source_concept
# and target_concept per pair. An empty target concept switches off
# localisation on the target side, which is what allows an empty target prompt.
CONCEPT="$CONCEPT" TARGET_CONCEPT="$TARGET_CONCEPT" \
SOURCE_PROMPT="$SOURCE_PROMPT" TARGET_PROMPT="$TARGET_PROMPT" \
EVAL_UNSEEN="${EVAL_UNSEEN:-}" PAIRS="$PAIRS" EVALS="$EVALS" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "benchmark")
from common import prompts_for

concept = os.environ["CONCEPT"]
source = os.environ["SOURCE_PROMPT"]

with open(os.environ["PAIRS"], "w", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "source_prompt": source,
        "target_prompt": os.environ["TARGET_PROMPT"],
        "source_concept": concept,
        "target_concept": os.environ["TARGET_CONCEPT"],
    }) + "\n")

# The unseen prompt is prompt 00 of v7, which is what the benchmark will score.
# For concepts outside Imagenette there is no v7 entry, so EVAL_UNSEEN has to
# supply one; without it the preview would just repeat the training prompt.
unseen = os.environ["EVAL_UNSEEN"]
if not unseen:
    try:
        unseen = prompts_for("v7", concept)[0]
    except SystemExit:
        unseen = ""

rows = [("train", source)]
if unseen and unseen != source:
    rows.append(("unseen", unseen))
with open(os.environ["EVALS"], "w", encoding="utf-8") as fh:
    for name, prompt in rows:
        fh.write(json.dumps({"name": name, "prompt": prompt}) + "\n")
PY

# PAIRS_FILE replaces the single generated pair with a hand-written set, which
# is how prompt augmentation is supplied. After the heredoc on purpose.
if [ -n "${PAIRS_FILE:-}" ]; then
    cp "$PAIRS_FILE" "$PAIRS"
    echo "pairs: copied from $PAIRS_FILE ($(wc -l < "$PAIRS") pairs)"
fi

echo "Host: $(hostname)"
echo "Task $IDX: $CONCEPT / $MAPPING"
echo "  source : $SOURCE_PROMPT"
echo "  target : ${TARGET_PROMPT:-(empty prompt)}"
echo "  target_concept: ${TARGET_CONCEPT:-(none)}"
echo "  lr $LEARNING_RATE   alpha $IMAP_ALPHA   mask $MASK_MODE   steps $MAX_TRAIN_STEPS"
echo "  skip_eval $SKIP_EVAL"
echo "  output : $OUTPUT_DIR"

CMD=(
    python train_hunyuan_imap_h_lora.py
    --model_path "${MODEL_PATH:-hunyuanvideo-community/HunyuanVideo}"
    --output_dir "$OUTPUT_DIR"
    --source_concept "$CONCEPT"
    --target_concept "$TARGET_CONCEPT"
    --prompt_pairs_jsonl "$PAIRS"
    --eval_prompts_jsonl "$EVALS"
    --eval_prompt "$SOURCE_PROMPT"
    --max_train_steps "$MAX_TRAIN_STEPS"
    --checkpointing_steps "$CHECKPOINTING_STEPS"
    --eval_steps "$EVAL_STEPS"
    --learning_rate "$LEARNING_RATE"
    --rank "$RANK" --lora_alpha "$LORA_ALPHA"
    --target_modules "$TARGET_MODULES"
    --imap_alpha "$IMAP_ALPHA" --imap_mode "$IMAP_MODE"
    --lamb_v "$LAMB_V" --v_steps "$V_STEPS" --v_block_min "$V_BLOCK_MIN"
    --v_early_prob "$V_EARLY_PROB" --v_mask_mode "$V_MASK_MODE"
    --v_video_only --no-v_disable_gc --split_hv_backward --gradient_checkpointing
    --double_blocks 1,2,4,5,6,9,10,13,15,16,17,18,19
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$TRAIN_NUM_FRAMES"
    --trajectory_steps 30
    --min_target_step "$MIN_TARGET_STEP" --max_target_step "$MAX_TARGET_STEP"
    --guidance_scale "$GUIDANCE_SCALE"
    --eval_height "$EVAL_HEIGHT" --eval_width "$EVAL_WIDTH"
    --eval_num_frames "$EVAL_NUM_FRAMES"
    --eval_inference_steps "$EVAL_INFERENCE_STEPS"
    --eval_seed "$EVAL_SEED" --seed "$SEED"
)

if [ "$SKIP_EVAL" = "1" ]; then CMD+=(--skip_eval); fi
# Unclamped means w = alpha*M, which for alpha>1 extrapolates past the target
# instead of landing on it: right for erasing a concept, wrong for replacing
# one object with another.
if [ "$MASK_MODE" = "clamped" ]; then CMD+=(--clamp_mask); fi

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
