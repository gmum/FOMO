#!/bin/bash
# =============================================================================
# One experiment -- one row of a table in the paper -- end to end.
#
#   bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
#   CONCEPT=church bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
#   DRY=1 bash slurm/reproduce_experiment.sh experiments/identities.yaml      print, submit nothing
#   SKIP_TRAIN=1 ...                                         adapters already exist
#   SKIP_BASE=1 ...                                          baseline already generated
#
# Five stages, chained by dependency:
#
#   1. the untouched model     keyed by the prompt set, so rows on the same
#                              prompts share it -- pass SKIP_BASE=1 for the
#                              second and later rows
#   2. training                one adapter per concept
#   3. the columns             every subject inside every column
#   4. scoring                 ResNet-50 or ArcFace
#   5. the numbers             prints the finished row into its log
#
# Dependencies are afterany, not afterok. A column whose training failed should
# leave a hole in one cell, not stop the other nine from being generated and
# scored; the hole is named when the numbers are printed.
# =============================================================================

set -euo pipefail

CONFIG="${1:?usage: bash slurm/reproduce_experiment.sh experiments/<name>.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
[ -f "$CONFIG" ] || { echo "ERROR: no such experiment: $CONFIG" >&2; exit 2; }

DRY="${DRY:-0}"
CONCEPT="${CONCEPT:-}"

eval "$(PYTHONPATH="$PWD" python3 -m experiments.config "$CONFIG" --shell)"
N_ROWS=$(echo "$EXP_ROWS" | wc -w)
N_SUBJECTS="$EXP_SUBJECTS"

# The command goes to stderr, because the job id is read back out of stdout.
run() {
    printf '  %s\n' "$*" >&2
    if [ "$DRY" = "1" ]; then echo "  [dry run, not submitted]" >&2; echo "DRY"; return 0; fi
    "$@"
}
submitted() { [ -n "${1:-}" ] && [ "$1" != "DRY" ]; }

# The position of one concept among this experiment's rows, and of one subject
# among its subjects. Empty when the name is not one of them.
index_of() {
    PYTHONPATH="$PWD" CONFIG="$CONFIG" WANTED="$CONCEPT" WHAT="$1" python3 - <<'PY'
import os
from experiments import config as ec
cfg = ec.load(os.environ["CONFIG"])
wanted, what = os.environ["WANTED"], os.environ["WHAT"]
names = ([r.concept for r in ec.rows(cfg)] if what == "row"
         else ec.subjects(cfg))
if wanted in names:
    print(names.index(wanted))
PY
}

ROW=""
SUBJECT=""
if [ -n "$CONCEPT" ]; then
    ROW=$(index_of row)
    SUBJECT=$(index_of subject)
    [ -n "$ROW" ] || { echo "ERROR: '$CONCEPT' is not a column of $EXP_NAME" >&2; exit 2; }
fi

echo "experiment : $EXP_NAME  ($EXP_KIND, method $EXP_METHOD)"
echo "rows       : $N_ROWS concepts x $N_SUBJECTS subjects"
[ -n "$CONCEPT" ] && echo "restricted : $CONCEPT  (row $ROW)"
echo ""

# --- 1. the untouched model -------------------------------------------------
BASE_ID=""
if [ "${SKIP_BASE:-0}" = "1" ]; then
    echo "baseline   : skipped on request"
elif [ "$EXP_METHOD" = "safree" ]; then
    # SAFREE replaces the pipeline class, so the untouched model is not
    # available in the same process. Generate it from a fomo row instead.
    echo "baseline   : skipped, SAFREE replaces the pipeline class"
else
    echo "baseline:"
    BASE_ID=$(run sbatch --parsable --array="${SUBJECT:-0-$((N_SUBJECTS - 1))}" \
        --export="ALL,CONFIG=$CONFIG,MODE=base" "$HERE/generate.sbatch")
fi
echo ""

# --- 2. training ------------------------------------------------------------
TRAIN_ID=""
if [ "$EXP_METHOD" != "fomo" ]; then
    echo "training   : nothing to train"
elif [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "training   : skipped on request"
else
    echo "training:"
    TRAIN_ID=$(run sbatch --parsable --array="${ROW:-0-$((N_ROWS - 1))}" \
        --export="ALL,CONFIG=$CONFIG" "$HERE/train.sbatch")
fi
echo ""

# --- 3. the columns ---------------------------------------------------------
GEN_DEP=()
STRICT_FLAG=""
if submitted "$TRAIN_ID"; then
    # STRICT because the training this waits on was supposed to produce the
    # weights. Without it a column whose training died would generate nothing
    # and report success, which looks exactly like a column that erased
    # everything.
    GEN_DEP=(--dependency="afterany:$TRAIN_ID" --kill-on-invalid-dep=yes)
    STRICT_FLAG=",STRICT=1"
fi
if [ -n "$ROW" ]; then
    GEN_ARRAY="$((ROW * N_SUBJECTS))-$((ROW * N_SUBJECTS + N_SUBJECTS - 1))"
else
    GEN_ARRAY="0-$((N_ROWS * N_SUBJECTS - 1))"
fi
echo "columns:"
COLS_ID=$(run sbatch --parsable "${GEN_DEP[@]}" --array="$GEN_ARRAY" \
    --export="ALL,CONFIG=$CONFIG$STRICT_FLAG" "$HERE/generate.sbatch")
echo ""

# --- 4. scoring -------------------------------------------------------------
DEPS=""
submitted "$BASE_ID" && DEPS="$BASE_ID"
submitted "$COLS_ID" && DEPS="${DEPS:+$DEPS:}$COLS_ID"
EVAL_DEP=()
[ -n "$DEPS" ] && EVAL_DEP=(--dependency="afterany:$DEPS" --kill-on-invalid-dep=yes)
if [ "$EXP_KIND" = "identities" ]; then
    EVAL_ARRAY="0"
elif [ -n "$ROW" ]; then
    EVAL_ARRAY="0,$((ROW + 1))"        # the baseline and that column
else
    EVAL_ARRAY="0-$N_ROWS"             # index 0 is the baseline
fi
echo "scoring:"
EVAL_ID=$(run sbatch --parsable "${EVAL_DEP[@]}" --array="$EVAL_ARRAY" \
    --export="ALL,CONFIG=$CONFIG" "$HERE/evaluate.sbatch")
echo ""

# --- 5. the numbers --------------------------------------------------------
REPORT_DEP=()
submitted "$EVAL_ID" && REPORT_DEP=(--dependency="afterany:$EVAL_ID" --kill-on-invalid-dep=yes)
echo "numbers:"
run sbatch --parsable "${REPORT_DEP[@]}" --array=0 \
    --export="ALL,CONFIG=$CONFIG,REPORT=1" "$HERE/evaluate.sbatch" > /dev/null
echo ""

[ "$DRY" = "1" ] || echo "squeue -u \$USER   to watch."
echo "the numbers land in logs/eval_*.out, and can be reprinted with:"
if [ "$EXP_KIND" = "identities" ]; then
    echo "  python3 -m evaluation.arcface $CONFIG --report"
else
    echo "  python3 -m evaluation.resnet $CONFIG --report"
fi
