#!/bin/bash
# =============================================================================
# The face-framing control: the same identity sweep on close-up prompts.
#
# In identity_prompts each person gets scenes that suit them, and a training
# pitch or a basketball court pulls the model toward the framing it saw those
# in -- a figure at distance. The face detection rate shows it: 100% for every
# politician, 97.8 for LeBron, 93.3 for Messi. Part of "the model cannot draw
# this person" was "the prompt did not put the face near the camera", and that
# is a property of the benchmark, not of the model.
#
# identity_prompts_face gives all nine the same thirty close-up templates, so
# framing stops being a per-person variable. Everything else -- weights, seed,
# resolution, mappings -- is unchanged, which is what makes this a control
# rather than a second experiment.
#
#   bash slurm/run_identity9_face.sh          base + ck10 + scoring
#   CKPT=000020 bash ...                      a different checkpoint
#   DRY=1 bash ...                            print the sbatch lines only
#
# Output: videos/identity9_face/, results/identity9_face/
# =============================================================================

set -euo pipefail

REPO_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning
cd "$REPO_DIR"

CKPT="${CKPT:-000010}"
TABLE="${TABLE:-configs/identity9_variants.tsv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos/identity9_face}"
PROMPTS_MODULE="${PROMPTS_MODULE:-identity_prompts_face}"
TAG="${TAG:-identity9_face}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
MASK_MODE="${MASK_MODE:-unclamped}"
IMAP_ALPHA="${IMAP_ALPHA:-7}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAPPINGS="${MAPPINGS:-null person demo}"
# IMAP_ALPHA picks the weights directory but does not appear in the method
# label, so two alpha settings would write videos over each other. The suffix
# is what keeps them apart, and it has to reach the scorer too.
METHOD_SUFFIX="${METHOD_SUFFIX:-}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
# Job ids the generation waits for, colon separated -- the training that has
# not finished yet. Setting it also turns on STRICT, so a task whose weights
# never arrived fails instead of writing an empty column.
AFTER="${AFTER:-}"
DRY="${DRY:-0}"

# The command goes to stderr, because the callers capture stdout to read the
# job id back out of --parsable.
run() {
    printf '  %s\n' "$*" >&2
    if [ "$DRY" = "1" ]; then echo "  [dry run, not submitted]" >&2; return 0; fi
    "$@"
}

LR_TAG="${LEARNING_RATE//[.-]/}"
# Probed only when not given. Once two sweeps hold the same alpha the glob
# stops being unambiguous, so anything scripted should pass SWEEP_DIR.
if [ -z "${SWEEP_DIR:-}" ]; then
    PROBE=$(ls -d outputs/*/donald_trump_demo_lr"$LR_TAG"/"$MASK_MODE"_alpha"$IMAP_ALPHA" 2>/dev/null | head -1 || true)
    [ -n "$PROBE" ] || { echo "ERROR: no trained identity weights under outputs/." >&2; exit 2; }
    SWEEP_DIR=$(echo "$PROBE" | cut -d/ -f2)
fi

N_VARIANTS=$(grep -c . "$TABLE")
N_PEOPLE=$(M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(len(importlib.import_module(os.environ["M"]).PEOPLE))')
N_TASKS=$((N_VARIANTS * N_PEOPLE))
# Counted from the table rather than by globbing the sweep, so that a filtered
# table -- one mapping instead of three -- checks its own nine variants and not
# all twenty-seven that happen to be on disk. Skipped when AFTER is set: the
# training that writes those checkpoints has not run yet, so instead the
# generation runs with STRICT=1 and fails loudly if they never appear.
if [ -n "$AFTER" ]; then
    HAVE=-1
else
HAVE=$(TABLE="$TABLE" SWEEP_DIR="$SWEEP_DIR" MASK_MODE="$MASK_MODE" \
       IMAP_ALPHA="$IMAP_ALPHA" LR_TAG="$LR_TAG" CKPT="$CKPT" python3 - <<'PY'
import os, sys
sys.path.insert(0, "benchmark")
from common import slugify

env = os.environ
found = 0
for line in open(env["TABLE"], encoding="utf-8"):
    if not line.strip():
        continue
    concept, mapping = line.rstrip("\n").split("\t")[:2]
    found += os.path.isdir(os.path.join(
        "outputs", env["SWEEP_DIR"],
        "%s_%s_lr%s" % (slugify(concept), mapping, env["LR_TAG"]),
        "%s_alpha%s" % (env["MASK_MODE"], env["IMAP_ALPHA"]),
        "checkpoint-%s" % env["CKPT"]))
print(found)
PY
)
fi

echo "sweep dir  : $SWEEP_DIR"
echo "prompts    : $PROMPTS_MODULE  ($N_PEOPLE people x $NUM_PROMPTS prompts)"
echo "videos     : $VIDEOS_ROOT"
if [ "$HAVE" = "-1" ]; then
    echo "checkpoint : $CKPT  (unchecked, waiting on job $AFTER)"
else
    echo "checkpoint : $CKPT  ($HAVE/$N_VARIANTS variants)"
fi
echo ""
[ "$HAVE" = "-1" ] || [ "$HAVE" -eq "$N_VARIANTS" ] || {
    echo "ERROR: checkpoint-$CKPT missing from some variants." >&2; exit 2; }

COMMON="TABLE=$TABLE,VIDEOS_ROOT=$VIDEOS_ROOT,SWEEP_DIR=$SWEEP_DIR,PROMPTS_MODULE=$PROMPTS_MODULE,NUM_PROMPTS=$NUM_PROMPTS,MASK_MODE=$MASK_MODE,IMAP_ALPHA=$IMAP_ALPHA,LEARNING_RATE=$LEARNING_RATE,METHOD_SUFFIX=$METHOD_SUFFIX,LORA_WEIGHT=$LORA_WEIGHT"
GEN_DEP=""
if [ -n "$AFTER" ]; then
    GEN_DEP="--dependency=afterany:$AFTER --kill-on-invalid-dep=yes"
    COMMON="$COMMON,STRICT=1"
fi

# --- baseline, one task per person ------------------------------------------
# The Original row has to be remeasured: it is the whole point. Comparing the
# old baseline against new close-up columns would report the framing change as
# an erasure result.
echo "baseline ($N_PEOPLE tasks):"
BASE=$(run sbatch --parsable --array=0-$((N_PEOPLE - 1)) \
    --export="ALL,MODE=base,$COMMON" slurm/gen_celeb_gh200.sbatch)
echo ""

# --- columns, one task per (variant, person) --------------------------------
echo "columns ($N_TASKS tasks):"
# shellcheck disable=SC2086
COLS=$(run sbatch --parsable $GEN_DEP --array=0-$((N_TASKS - 1)) \
    --export="ALL,MODE=column,$COMMON,CHECKPOINT=checkpoint-$CKPT" \
    slurm/gen_celeb_gh200.sbatch)
echo ""

# --- scoring, behind both ----------------------------------------------------
METHODS=""
for MAP in $MAPPINGS; do
    METHODS="${METHODS:+$METHODS }imap_${MAP}_lr${LR_TAG}${METHOD_SUFFIX}_${CKPT}"
done
N_METHODS=$(echo "$METHODS" | wc -w)

echo "scoring ($N_METHODS method labels):"
# The scoring sbatch only creates results/identity9, so the directory for a
# different TAG has to exist before the job starts.
mkdir -p "results/$TAG"
DEP=""
if [ "$DRY" != "1" ]; then
    DEP="--dependency=afterany:$BASE:$COLS --kill-on-invalid-dep=yes"
fi
# shellcheck disable=SC2086
SCORE=$(run sbatch --parsable $DEP --array=0-$((N_METHODS - 1)) \
    --export="ALL,VIDEOS_ROOT=$VIDEOS_ROOT,TAG=$TAG,PROMPTS_MODULE=$PROMPTS_MODULE,METHODS=$METHODS" \
    slurm/id_similarity_split_gh200.sbatch)
echo ""

if [ "$DRY" != "1" ]; then
    echo "baseline $BASE   columns $COLS   scoring $SCORE (waits for both)"
fi
echo "squeue -u \$USER   to watch."
