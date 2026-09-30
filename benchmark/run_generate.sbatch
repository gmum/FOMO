#!/bin/bash
# =============================================================================
# Generate evaluation videos as a SLURM job array.
#
# The array index selects the work unit, and what that unit is depends on the
# method:
#
#   MODE=column     index = class (0..9).  One column, ten tasks.
#                   Used for the base model and for LoRA methods, where the
#                   column is fixed by ERASED / MODELS.
#
#   MODE=matrix     index = column*10 + class (0..99).  Full 10x10 matrix.
#                   Used for negprompt, where every column needs all ten
#                   classes generated under that column's negative prompt,
#                   otherwise PSR cannot be measured.
#
# Examples
# --------
# Base model, all ten classes:
#   sbatch --array=0-9%5 --export=ALL,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720 \
#          benchmark/run_generate.sbatch
#
# CFG-matched control (empty negative prompt), lands next to base:
#   sbatch --array=0-9%5 --export=ALL,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,\
# NEGPROMPT=1,NEGPROMPT_TEXT=,NEGPROMPT_LABEL=cfgbase benchmark/run_generate.sbatch
#
# SAFREE, full matrix (every column erases its own concept):
#   sbatch --array=0-99%10 --export=ALL,MODE=matrix,PROMPT_SET=v7,WIDTH=1280,HEIGHT=720,\
# SAFREE=1,NO_BASE=1 benchmark/run_generate.sbatch
#
# NegPrompt, full matrix:
#   sbatch --array=0-99%10 --export=ALL,MODE=matrix,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,\
# NEGPROMPT=1 benchmark/run_generate.sbatch
#
# A trained method, one column:
#   sbatch --array=0-9%5 --export=ALL,PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,\
# ERASED="church",MODELS="ours=outputs/imap_in_church_to_house/checkpoint-000150",\
# NO_BASE=1 benchmark/run_generate.sbatch
#
# Resumable: --skip_existing is on by default, so re-running fills gaps only.
# =============================================================================

#SBATCH --job-name=erase_gen
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=01:15:00
#SBATCH --output=logs/erase_gen_%A_%a.out
#SBATCH --error=logs/erase_gen_%A_%a.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$TMPDIR"

# Activates the conda environment or ends the job with a readable message.
# shellcheck source=benchmark/env.sh
source benchmark/env.sh
require_modules torch diffusers

CLASS_SET="${CLASS_SET:-imagenette}"
MODEL_FAMILY="${MODEL_FAMILY:-hunyuan}"
PROMPT_SET="${PROMPT_SET:-v1}"
MODE="${MODE:-column}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-288}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SEED="${SEED:-42}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-$GUIDANCE_SCALE}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos}"
SHARE_BASE="${SHARE_BASE:-1}"

# Class list and directory naming come from benchmark/common.py, so generation
# and evaluation can never disagree about where files live. Everything is
# resolved in one call, with values passed through the environment rather than
# interpolated into the source — class names contain spaces.
IDX="${SLURM_ARRAY_TASK_ID:-0}"
export CLASS_SET PROMPT_SET MODE WIDTH HEIGHT GUIDANCE_SCALE SEED IDX MODEL_FAMILY
export ERASED="${ERASED:-}"

RESOLVED="$(python3 - <<'PY'
import os, sys
sys.path.insert(0, "benchmark")
from common import CLASS_SETS, set_tag, column_dir, slugify

classes = CLASS_SETS[os.environ["CLASS_SET"]]
n = len(classes)
idx = int(os.environ["IDX"])
mode = os.environ["MODE"]

if mode == "matrix":
    limit = n * n
    col_idx, cls_idx = divmod(idx, n)
else:
    limit = n
    col_idx, cls_idx = None, idx

if idx >= limit:
    sys.exit("ERROR: array index %d out of range 0..%d for MODE=%s"
             % (idx, limit - 1, mode))

concept = classes[cls_idx]
erased = classes[col_idx] if mode == "matrix" else (os.environ["ERASED"] or None)

tag = set_tag(os.environ["CLASS_SET"], os.environ["PROMPT_SET"],
              int(os.environ["WIDTH"]), int(os.environ["HEIGHT"]),
              float(os.environ["GUIDANCE_SCALE"]), int(os.environ["SEED"]),
              model=os.environ["MODEL_FAMILY"])

# One field per line: concept, erased, set tag, class slug.
print(concept)
print(erased if erased else "")
print(tag)
print(slugify(concept))
print(column_dir(erased))
PY
)" || { echo "$RESOLVED"; exit 1; }

CONCEPT="$(sed -n 1p <<< "$RESOLVED")"
ERASED="$(sed -n 2p <<< "$RESOLVED")"
SET_TAG="$(sed -n 3p <<< "$RESOLVED")"
CLASS_SLUG="$(sed -n 4p <<< "$RESOLVED")"
COLUMN="$(sed -n 5p <<< "$RESOLVED")"

OUTPUT_DIR="${OUTPUT_DIR:-$VIDEOS_ROOT/$SET_TAG/$COLUMN}"
SHARED_BASE_DIR="$VIDEOS_ROOT/$SET_TAG/_shared/base/$CLASS_SLUG"
COLUMN_BASE_DIR="$OUTPUT_DIR/$CLASS_SLUG/base"
SHARED_REL="../../_shared/base/$CLASS_SLUG"

echo "Task $IDX | mode=$MODE | column='${ERASED:-none}' | generating '$CONCEPT'"
echo "  host   : $(hostname)  $(date)"
echo "  output : $OUTPUT_DIR/$CLASS_SLUG/"

# The base model does not depend on which concept is erased: same prompts,
# same seeds, no intervention. Keep one copy under _shared/base/<class> and
# symlink it into every column, instead of regenerating it per column.
NO_BASE="${NO_BASE:-}"
if [ "$SHARE_BASE" = "1" ] && [ -z "$NO_BASE" ]; then
    if compgen -G "$SHARED_BASE_DIR/*.mp4" > /dev/null 2>&1; then
        mkdir -p "$(dirname "$COLUMN_BASE_DIR")"
        ln -sfn "$SHARED_REL" "$COLUMN_BASE_DIR"
        echo "  base   : reusing $SHARED_BASE_DIR (symlinked, generation skipped)"
        NO_BASE=1
    fi
fi

CMD=(python3 benchmark/generate.py
    --concepts "$CONCEPT"
    --output_dir "$OUTPUT_DIR"
    --class_set "$CLASS_SET"
    --prompt_set "$PROMPT_SET"
    --num_prompts "$NUM_PROMPTS"
    --model_family "$MODEL_FAMILY"
    --height "$HEIGHT" --width "$WIDTH"
    --num_frames "$NUM_FRAMES" --fps "$FPS"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --seed "$SEED"
    --lora_weight "$LORA_WEIGHT")

[ -n "${MODEL_PATH:-}" ] && CMD+=(--model_path "$MODEL_PATH")
[ -n "$NO_BASE" ] && CMD+=(--no_base)

if [ -n "${MODELS:-}" ]; then
    # shellcheck disable=SC2206
    MODEL_ARR=($MODELS)
    CMD+=(--models "${MODEL_ARR[@]}")
fi

if [ -n "${SAFREE:-}" ]; then
    # SAFREE replaces the pipeline class, so it runs alone. It samples like the
    # base model (do_true_cfg is hard-coded off), so it needs no CFG control.
    CMD+=(--safree --safree_label "${SAFREE_LABEL:-safree}")
fi

if [ -n "${NEGPROMPT:-}" ]; then
    CMD+=(--negprompt --true_cfg_scale "$TRUE_CFG_SCALE"
          --negprompt_label "${NEGPROMPT_LABEL:-negprompt}")
    # NEGPROMPT_TEXT unset  -> negate the column's concept
    # NEGPROMPT_TEXT=""     -> CFG-matched control
    if [ "${NEGPROMPT_TEXT+set}" = "set" ]; then
        CMD+=(--negprompt_text "$NEGPROMPT_TEXT")
    else
        CMD+=(--negprompt_text "$ERASED")
    fi
fi

echo "Command:"
printf '  %q' "${CMD[@]}"; echo
echo "----------------------------------------------------------"
"${CMD[@]}"

# Move a freshly generated base into _shared and leave a symlink behind, so the
# next column reuses it. The [ ! -L ] guard keeps this idempotent.
if [ "$SHARE_BASE" = "1" ] \
   && [ -d "$COLUMN_BASE_DIR" ] && [ ! -L "$COLUMN_BASE_DIR" ]; then
    mkdir -p "$(dirname "$SHARED_BASE_DIR")"
    if [ -e "$SHARED_BASE_DIR" ]; then
        echo "base: $SHARED_BASE_DIR already exists — leaving the column copy alone"
    else
        mv "$COLUMN_BASE_DIR" "$SHARED_BASE_DIR"
        ln -sfn "$SHARED_REL" "$COLUMN_BASE_DIR"
        echo "base: moved to $SHARED_BASE_DIR and symlinked"
    fi
fi

echo "Task $IDX done | $(date)"
