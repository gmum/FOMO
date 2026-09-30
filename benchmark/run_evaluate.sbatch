#!/bin/bash
# =============================================================================
# Evaluate generated videos as a SLURM job array.
#
# Two modes, matching the two ways a table row is built:
#
#   MODE=shared      one task. Scores the baseline column (unlearn_none),
#                    which contains every method whose videos do not depend on
#                    the column: base and the CFG-matched control.
#                    Results -> results/<tag>/per_class.csv
#
#   MODE=per-column  ten tasks, index = column (0..9). Scores one column each,
#                    for methods whose videos differ per column.
#                    Results -> results/<tag>/<method>/<class>/per_class.csv
#
# Examples
# --------
# Baseline and CFG control in one pass:
#   sbatch --array=0 --export=ALL,MODE=shared,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,\
# METHODS="base cfgbase" benchmark/run_evaluate.sbatch
#
# NegPrompt, all ten columns:
#   sbatch --array=0-9%5 --export=ALL,MODE=per-column,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,\
# METHOD=negprompt benchmark/run_evaluate.sbatch
#
# Then build the row:
#   python3 benchmark/make_row.py --mode shared \
#       --per_class results/imagenette_v1_1280x720/per_class.csv \
#       --class_set imagenette --method base --label "HunyuanVideo"
#
#   python3 benchmark/make_row.py --mode per-column \
#       --results_root results/imagenette_v1_1280x720/negprompt \
#       --class_set imagenette --method negprompt --label "NegPrompt"
#
# Cheap: a few minutes per task. Safe to re-run; results are overwritten.
# =============================================================================

#SBATCH --job-name=erase_eval
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=logs/erase_eval_%A_%a.out
#SBATCH --error=logs/erase_eval_%A_%a.err

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
mkdir -p logs "$TORCH_HOME" "$TMPDIR"

# Activates the conda environment or ends the job with a readable message.
# shellcheck source=benchmark/env.sh
source benchmark/env.sh

CLASS_SET="${CLASS_SET:-imagenette}"
MODEL_FAMILY="${MODEL_FAMILY:-hunyuan}"
PROMPT_SET="${PROMPT_SET:-v1}"
MODE="${MODE:-shared}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-288}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SEED="${SEED:-42}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CLASSIFIER="${CLASSIFIER:-resnet50}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

# The two class sets use different backends: torchvision for the ImageNet
# classifier, transformers for CLIP zero-shot.
if [ "$CLASS_SET" = "imagenette" ]; then
    require_modules torch torchvision
else
    require_modules torch transformers
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
export CLASS_SET PROMPT_SET MODE WIDTH HEIGHT GUIDANCE_SCALE SEED IDX MODEL_FAMILY

RESOLVED="$(python3 - <<'PY'
import os, sys
sys.path.insert(0, "benchmark")
from common import CLASS_SETS, set_tag, column_dir, slugify

classes = CLASS_SETS[os.environ["CLASS_SET"]]
idx = int(os.environ["IDX"])
mode = os.environ["MODE"]

if mode == "shared":
    if idx != 0:
        sys.exit("ERROR: MODE=shared takes a single task, use --array=0")
    erased = classes[0]          # only labels summary.json; the row covers all ten
    column = column_dir(None)
else:
    if idx >= len(classes):
        sys.exit("ERROR: array index %d out of range 0..%d"
                 % (idx, len(classes) - 1))
    erased = classes[idx]
    column = column_dir(erased)

tag = set_tag(os.environ["CLASS_SET"], os.environ["PROMPT_SET"],
              int(os.environ["WIDTH"]), int(os.environ["HEIGHT"]),
              float(os.environ["GUIDANCE_SCALE"]), int(os.environ["SEED"]),
              model=os.environ["MODEL_FAMILY"])

print(erased)
print(tag)
print(column)
print(slugify(erased))
PY
)" || { echo "$RESOLVED"; exit 1; }

ERASED="$(sed -n 1p <<< "$RESOLVED")"
SET_TAG="$(sed -n 2p <<< "$RESOLVED")"
COLUMN="$(sed -n 3p <<< "$RESOLVED")"
ERASED_SLUG="$(sed -n 4p <<< "$RESOLVED")"

VIDEOS_DIR="${VIDEOS_DIR:-$VIDEOS_ROOT/$SET_TAG/$COLUMN}"
if [ "$MODE" = "shared" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$SET_TAG}"
    METHODS="${METHODS:-base}"
else
    METHOD="${METHOD:-negprompt}"
    OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$SET_TAG/$METHOD/$ERASED_SLUG}"
    METHODS="${METHODS:-$METHOD}"
fi

echo "Task $IDX | mode=$MODE | column='$ERASED'"
echo "  videos : $VIDEOS_DIR"
echo "  results: $OUTPUT_DIR"

# A missing directory is not an error: some configurations may still be
# generating. Exit zero so one unfinished entry does not mark the whole array
# as FAILED — gaps show up later when the row is assembled.
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "  SKIPPED: directory does not exist (not generated yet?)"
    exit 0
fi
N_MP4=$(find -L "$VIDEOS_DIR" -name '*.mp4' | wc -l)
echo "  found  : $N_MP4 mp4"
if [ "$N_MP4" -eq 0 ]; then
    echo "  SKIPPED: no videos"
    exit 0
fi

# shellcheck disable=SC2206
METHOD_ARR=($METHODS)

CMD=(python3 benchmark/evaluate.py
    --videos_dir "$VIDEOS_DIR"
    --output_dir "$OUTPUT_DIR"
    --class_set "$CLASS_SET"
    --erased_concept "$ERASED"
    --methods "${METHOD_ARR[@]}"
    --batch_size "$BATCH_SIZE")

if [ "$CLASS_SET" = "imagenette" ]; then
    CMD+=(--classifier "$CLASSIFIER")
else
    CMD+=(--clip_model "$CLIP_MODEL")
fi

if [ -n "${PROMPT_IDS:-}" ]; then
    # shellcheck disable=SC2206
    PROMPT_ARR=($PROMPT_IDS)
    CMD+=(--prompt_ids "${PROMPT_ARR[@]}")
fi

echo "Command:"
printf '  %q' "${CMD[@]}"; echo
echo "----------------------------------------------------------"
"${CMD[@]}"

echo "Task $IDX done | $(date)"
