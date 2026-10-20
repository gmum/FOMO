#!/bin/bash
# =============================================================================
# Score what was generated.
#
# Objects, one task per column plus one for the baseline:
#
#   sbatch --array=0-10 --export=ALL,CONFIG=experiments/objects_mapping2.yaml \
#          slurm/evaluate.sbatch
#
#   Index 0 is the untouched model, which is why the array is one longer than
#   the number of columns. It is scored once per prompt set, not once per
#   experiment, so re-running it for a second mapping costs nothing and
#   overwrites the same numbers.
#
# Identities, a single task over the whole matrix:
#
#   sbatch --array=0 --export=ALL,CONFIG=experiments/identities.yaml \
#          slurm/evaluate.sbatch
#
#   One task because the ArcFace side is cheap next to generation, and
#   --resume makes a job killed at the wall clock cost nothing: resubmit and it
#   carries on from the per-video CSV.
#
# Both are safe to re-run; results are overwritten.
# =============================================================================

#SBATCH --job-name=fomo_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G

set -euo pipefail

# shellcheck source=slurm/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${CONFIG:?set CONFIG=experiments/<name>.yaml}"
load_experiment "$CONFIG"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# REPORT=1 scores nothing: it reads back what the scoring jobs wrote and prints
# the finished rows, so the numbers end up in a log rather than waiting to be
# asked for. Needs no GPU and no heavy imports.
if [ "${REPORT:-0}" = "1" ]; then
    if [ "$EXP_KIND" = "identities" ]; then
        exec python3 -m evaluation.arcface "$CONFIG" --report
    fi
    exec python3 -m evaluation.resnet "$CONFIG" --report
fi

if [ "$EXP_KIND" = "identities" ]; then
    require_modules insightface onnxruntime
    echo "task $IDX | ArcFace over $EXP_NAME"
    exec python3 -m evaluation.arcface "$CONFIG" --resume \
        --refs-dir "${REFS_DIR:-data/identity_refs}"
fi

require_modules torch torchvision
if [ "$IDX" = "0" ]; then
    echo "task $IDX | baseline for prompt set '$EXP_PROMPTS'"
    exec python3 -m evaluation.resnet "$CONFIG" --base
fi

ERASED=$(python3 -m experiments.config "$CONFIG" --row "$((IDX - 1))" | sed -n 1p)
echo "task $IDX | column '$ERASED'"
exec python3 -m evaluation.resnet "$CONFIG" --erased "$ERASED"
