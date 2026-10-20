#!/bin/bash
# =============================================================================
# Generate one cell: one subject inside one column.
#
#   MODE=base       index = subject.
#                   The untouched model, which every experiment on the same
#                   prompt set shares. Ten tasks for objects, five for people.
#
#   MODE=column     index = row * subjects + subject.   (the default)
#                   Every column generates every subject, which is what makes
#                   the off-diagonal measurable: PSR for objects, Preserve for
#                   identities.
#
#   sbatch --array=0-9  --export=ALL,CONFIG=experiments/identities.yaml,MODE=base \
#          slurm/generate.sbatch
#   sbatch --array=0-99 --export=ALL,CONFIG=experiments/objects_mapping2.yaml \
#          slurm/generate.sbatch
#
# `python3 -m experiments.config <config>` prints how many rows there are, so
# the array bound is rows * subjects - 1 for a column run.
#
# Columns with no trained adapter exit quietly, so the array can be submitted
# as a dependency and re-run later to fill gaps. STRICT=1 turns that into a
# failure, for when the weights are supposed to be there already: a column that
# silently generates nothing looks exactly like a column that erased everything.
# =============================================================================

#SBATCH --job-name=fomo_gen
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

set -euo pipefail

# shellcheck source=slurm/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_modules torch diffusers

: "${CONFIG:?set CONFIG=experiments/<name>.yaml}"
load_experiment "$CONFIG"

MODE="${MODE:-column}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
N="$EXP_SUBJECTS"

CMD=(python3 -m experiments.generate "$CONFIG")

if [ "$MODE" = "base" ]; then
    SUBJECT=$(python3 -m experiments.config "$CONFIG" --subject-at "$IDX")
    CMD+=(--base --subject "$SUBJECT")
    echo "task $IDX | base | generating '$SUBJECT'"
else
    ROW_IDX=$((IDX / N))
    SUB_IDX=$((IDX % N))
    ERASED=$(python3 -m experiments.config "$CONFIG" --row "$ROW_IDX" | sed -n 1p)
    SUBJECT=$(python3 -m experiments.config "$CONFIG" --subject-at "$SUB_IDX")
    CMD+=(--erased "$ERASED" --subject "$SUBJECT")
    [ "${STRICT:-0}" = "1" ] || CMD+=(--allow-missing)
    echo "task $IDX | column '$ERASED' | generating '$SUBJECT'"
fi

[ -n "${LORA_WEIGHT:-}" ] && CMD+=(--lora-weight "$LORA_WEIGHT")

printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
