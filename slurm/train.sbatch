#!/bin/bash
# =============================================================================
# Train one adapter: one concept of one experiment.
#
#   sbatch --array=0-9 --export=ALL,CONFIG=experiments/objects_mapping2.yaml \
#          slurm/train.sbatch
#
# The array index is the position of the row inside the experiment, counted
# from zero -- not the line number of the whole mapping table. An experiment
# selecting ten of thirty rows therefore wants --array=0-9, and
# `python3 -m experiments.config <config>` prints which rows those are.
#
# Weights: adapters/<experiment>/<concept>/<mask>_alpha<alpha>/checkpoint-*
#
# Partition and account are the two site-specific lines in this file. Everything
# else comes from the experiment.
# =============================================================================

#SBATCH --job-name=fomo_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

# shellcheck source=slurm/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_modules torch diffusers peft

: "${CONFIG:?set CONFIG=experiments/<name>.yaml}"
load_experiment "$CONFIG"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
mapfile -t ROW < <(python3 -m experiments.config "$CONFIG" --row "$IDX")
CONCEPT="${ROW[0]}"
TARGET_CONCEPT="${ROW[1]}"
SOURCE_PROMPT="${ROW[2]}"
TARGET_PROMPT="${ROW[3]}"
UNSEEN_PROMPT="${ROW[4]}"

SLUG=$(python3 -c 'import sys; from experiments.layout import slugify; print(slugify(sys.argv[1]))' "$CONCEPT")
OUTPUT_DIR="${ADAPTERS_ROOT:-adapters}/$EXP_NAME/$SLUG/${TRAIN_MASK_MODE}_alpha${TRAIN_IMAP_ALPHA}"
mkdir -p "$OUTPUT_DIR"

# The concepts travel in the pair file, because the trainer reads
# source_concept and target_concept per pair. An empty target concept switches
# off localisation on the target side, which is what lets the target prompt be
# empty too -- that is the null mapping.
PAIRS="$OUTPUT_DIR/pairs.jsonl"
python3 - "$PAIRS" "$SOURCE_PROMPT" "$TARGET_PROMPT" "$CONCEPT" "$TARGET_CONCEPT" <<'PY'
import json, sys
path, source, target, concept, target_concept = sys.argv[1:6]
with open(path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"source_prompt": source, "target_prompt": target,
                         "source_concept": concept,
                         "target_concept": target_concept}) + "\n")
PY

# Previews during training, if they are switched on: the prompt that was trained
# on, and prompt 00 of the evaluation set, which is what the benchmark scores.
# Seeing only the training prompt tells you nothing about generalisation.
EVALS="$OUTPUT_DIR/eval_prompts.jsonl"
python3 - "$EVALS" "$SOURCE_PROMPT" "$UNSEEN_PROMPT" <<'PY'
import json, sys
path, train, unseen = sys.argv[1:4]
rows = [("train", train)]
if unseen and unseen != train:
    rows.append(("unseen", unseen))
with open(path, "w", encoding="utf-8") as fh:
    for name, prompt in rows:
        fh.write(json.dumps({"name": name, "prompt": prompt}) + "\n")
PY

echo "host    : $(hostname)"
echo "task $IDX: $CONCEPT -> ${TARGET_CONCEPT:-(nothing)}"
echo "  source: $SOURCE_PROMPT"
echo "  target: ${TARGET_PROMPT:-(empty prompt)}"
echo "  steps $TRAIN_MAX_TRAIN_STEPS   lr $TRAIN_LEARNING_RATE   alpha $TRAIN_IMAP_ALPHA"
echo "  output: $OUTPUT_DIR"

CMD=(
    python3 -m fomo.train
    --model_path "$GEN_MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --source_concept "$CONCEPT"
    --target_concept "$TARGET_CONCEPT"
    --prompt_pairs_jsonl "$PAIRS"
    --eval_prompts_jsonl "$EVALS"
    --eval_prompt "$SOURCE_PROMPT"
    --max_train_steps "$TRAIN_MAX_TRAIN_STEPS"
    --checkpointing_steps "$TRAIN_CHECKPOINTING_STEPS"
    --learning_rate "$TRAIN_LEARNING_RATE"
    --rank "$TRAIN_RANK" --lora_alpha "$TRAIN_LORA_ALPHA"
    --target_modules "$TRAIN_TARGET_MODULES"
    --imap_alpha "$TRAIN_IMAP_ALPHA"
    --lamb_v "$TRAIN_LAMB_V" --v_steps "$TRAIN_V_STEPS"
    --v_block_min "$TRAIN_V_BLOCK_MIN" --v_early_prob "$TRAIN_V_EARLY_PROB"
    --v_mask_mode "$TRAIN_V_MASK_MODE"
    --double_blocks "$TRAIN_DOUBLE_BLOCKS"
    --trajectory_steps "$TRAIN_TRAJECTORY_STEPS"
    --height "$GEN_HEIGHT" --width "$GEN_WIDTH" --num_frames "$GEN_NUM_FRAMES"
    --guidance_scale "$GEN_GUIDANCE_SCALE"
    --seed "$GEN_SEED"
    # Gradient checkpointing has to be on for both passes. --v_disable_gc
    # defaults to true and turns it off during the preservation pass, which
    # fits at 512x288 and runs out of memory at 1280x720.
    --v_video_only --no-v_disable_gc --split_hv_backward --gradient_checkpointing
    --skip_eval
)
if [ "$TRAIN_MASK_MODE" = "clamped" ]; then CMD+=(--clamp_mask); fi

printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
