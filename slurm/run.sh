#!/bin/bash
# =============================================================================
# One experiment end to end: train, generate, score. Four submissions, chained
# by dependency, so the whole thing can be launched and left alone.
#
#   bash slurm/run.sh experiments/objects_mapping2.yaml
#   DRY=1 bash slurm/run.sh experiments/identities.yaml      print, submit nothing
#   SKIP_TRAIN=1 bash slurm/run.sh ...                       adapters already exist
#   SKIP_BASE=1 bash slurm/run.sh ...                        baseline already generated
#
# The baseline is keyed by the prompt set, so running a second mapping on the
# same prompts should pass SKIP_BASE=1 -- the videos are already there and the
# job would skip every file anyway, but at the cost of queueing for a GPU.
#
# Dependencies are afterany rather than afterok on purpose. A column whose
# training failed should leave a hole in one cell, not stop the other nine from
# being generated and scored; the hole is reported when the table is assembled.
# =============================================================================

set -euo pipefail

CONFIG="${1:?usage: bash slurm/run.sh experiments/<name>.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

DRY="${DRY:-0}"
eval "$(PYTHONPATH="$PWD" python3 -m experiments.config "$CONFIG" --shell)"

N_ROWS=$(echo "$EXP_ROWS" | wc -w)
N_SUBJECTS="$EXP_SUBJECTS"
N_CELLS=$((N_ROWS * N_SUBJECTS))

# The command goes to stderr, because the job id is read back out of stdout.
run() {
    printf '  %s\n' "$*" >&2
    if [ "$DRY" = "1" ]; then echo "  [dry run, not submitted]" >&2; echo "DRY"; return 0; fi
    "$@"
}

echo "experiment : $EXP_NAME  ($EXP_KIND, method $EXP_METHOD)"
echo "rows       : $N_ROWS concepts x $N_SUBJECTS subjects = $N_CELLS cells"
echo "prompts    : $EXP_PROMPTS"
echo ""

DEPS=""
add_dep() { [ "$1" = "DRY" ] || DEPS="${DEPS:+$DEPS:}$1"; }

# --- training ----------------------------------------------------------------
TRAIN_ID=""
if [ "$EXP_METHOD" != "fomo" ]; then
    echo "training   : skipped, method '$EXP_METHOD' trains nothing"
elif [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "training   : skipped on request"
else
    echo "training ($N_ROWS tasks):"
    TRAIN_ID=$(run sbatch --parsable --array=0-$((N_ROWS - 1)) \
        --export="ALL,CONFIG=$CONFIG" "$HERE/train.sbatch")
fi
echo ""

# --- the untouched model -----------------------------------------------------
if [ "$EXP_METHOD" = "safree" ]; then
    echo "baseline   : skipped, SAFREE replaces the pipeline class"
elif [ "${SKIP_BASE:-0}" = "1" ]; then
    echo "baseline   : skipped on request"
else
    echo "baseline ($N_SUBJECTS tasks):"
    BASE_ID=$(run sbatch --parsable --array=0-$((N_SUBJECTS - 1)) \
        --export="ALL,CONFIG=$CONFIG,MODE=base" "$HERE/generate.sbatch")
    add_dep "$BASE_ID"
fi
echo ""

# --- the columns -------------------------------------------------------------
echo "columns ($N_CELLS tasks):"
GEN_DEP=()
if [ -n "$TRAIN_ID" ] && [ "$TRAIN_ID" != "DRY" ]; then
    # STRICT because the training this waits on was supposed to produce the
    # weights. Without it a column whose training died would generate nothing
    # and report success.
    GEN_DEP=(--dependency="afterany:$TRAIN_ID" --kill-on-invalid-dep=yes)
    STRICT_FLAG=",STRICT=1"
else
    STRICT_FLAG=""
fi
COLS_ID=$(run sbatch --parsable "${GEN_DEP[@]}" --array=0-$((N_CELLS - 1)) \
    --export="ALL,CONFIG=$CONFIG$STRICT_FLAG" "$HERE/generate.sbatch")
add_dep "$COLS_ID"
echo ""

# --- scoring ----------------------------------------------------------------
if [ "$EXP_KIND" = "identities" ]; then
    EVAL_ARRAY="0"
else
    EVAL_ARRAY="0-$N_ROWS"          # index 0 is the baseline
fi
echo "scoring:"
EVAL_DEP=()
[ -n "$DEPS" ] && EVAL_DEP=(--dependency="afterany:$DEPS" --kill-on-invalid-dep=yes)
EVAL_ID=$(run sbatch --parsable "${EVAL_DEP[@]}" --array="$EVAL_ARRAY" \
    --export="ALL,CONFIG=$CONFIG" "$HERE/evaluate.sbatch")
echo ""

if [ "$DRY" != "1" ]; then
    echo "train ${TRAIN_ID:-none}   columns $COLS_ID   scoring $EVAL_ID"
    echo "squeue -u \$USER   to watch."
fi
echo ""
echo "then, once scoring is done:"
if [ "$EXP_KIND" = "identities" ]; then
    echo "  python3 -m evaluation.tables $CONFIG --out tables/$EXP_NAME.tex"
else
    echo "  python3 -m evaluation.tables experiments/objects_null.yaml \\"
    echo "      experiments/objects_mapping1.yaml experiments/objects_mapping2.yaml \\"
    echo "      --out tables/objects.tex"
fi
