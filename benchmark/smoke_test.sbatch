#!/bin/bash
# =============================================================================
# End-to-end smoke test: generate -> evaluate -> row, on a throwaway directory.
#
#   sbatch benchmark/smoke_test.sbatch
#
# Two classes, three prompts each, 512x288. Around ten minutes, most of it
# spent loading the model. Everything goes under smoke/, which can be deleted
# afterwards; nothing touches videos/ or results/.
#
# Run this BEFORE deleting the old scripts. It exercises the real pipeline,
# the real classifier and the real argument parsing, which is what unit-level
# checks cannot cover.
# =============================================================================

#SBATCH --job-name=erase_smoke
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/erase_smoke_%j.out
#SBATCH --error=logs/erase_smoke_%j.err

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
require_modules torch torchvision diffusers

OUT=smoke
rm -rf "$OUT"
mkdir -p "$OUT"

echo "############ 1/5  imports and prompt rendering (no GPU) ############"
python3 - <<'PY'
import sys
sys.path.insert(0, "benchmark")
from common import CLASS_SETS, PROMPT_SETS, render, set_tag, column_dir, slugify
print("class sets      :", ", ".join(sorted(CLASS_SETS)))
print("prompt sets     :", ", ".join("%s(%d)" % (k, len(v)) for k, v in sorted(PROMPT_SETS.items())))
print("example prompt  :", render(PROMPT_SETS["v1"][0], "English springer"))
print("tag             :", set_tag("imagenette", "v1", 1280, 720, 6.0, 42))
print("column          :", column_dir("garbage truck"))
print("slug            :", slugify("English springer"))
PY

echo
echo "############ 2/5  dry run: prompts and paths, still no GPU ############"
python3 benchmark/generate.py \
    --concepts "church" --output_dir "$OUT/videos/tag/unlearn_none" \
    --class_set imagenette --prompt_set v1 --num_prompts 3 --dry_run

echo
echo "############ 3/5  generate 2 classes x 3 prompts, base + negprompt ############"
python3 benchmark/generate.py \
    --concepts "church" "golf ball" \
    --output_dir "$OUT/videos/tag/unlearn_church" \
    --class_set imagenette --prompt_set v1 --num_prompts 3 \
    --height 288 --width 512 --num_frames 17 \
    --num_inference_steps 10 \
    --negprompt --negprompt_text "church" --true_cfg_scale 6.0

echo
echo "generated files:"
find "$OUT/videos" -name '*.mp4' | sort

echo
echo "############ 4/5  evaluate ############"
python3 benchmark/evaluate.py \
    --videos_dir "$OUT/videos/tag/unlearn_church" \
    --output_dir "$OUT/results" \
    --class_set imagenette --erased_concept "church" \
    --classifier resnet50 --batch_size 32

echo
echo "############ 5/5  build a row ############"
python3 benchmark/make_row.py --mode shared \
    --per_class "$OUT/results/per_class.csv" \
    --class_set imagenette --method base --label "smoke" \
    --out_json "$OUT/results/row.json"

echo
echo "=========================================================="
echo "SMOKE TEST PASSED"
echo "Only two of ten classes were generated, so the row is"
echo "incomplete by design — what matters is that every stage ran."
echo "Remove with:  rm -rf $OUT"
echo "=========================================================="
