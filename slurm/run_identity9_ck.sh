#!/bin/bash
# =============================================================================
# Earlier checkpoints for the identity table, end to end.
#
# Preserve rises the earlier the checkpoint is stopped -- ck30 beats ck50 in
# every mapping -- while Erase is already at the noise floor by ck30. That
# leaves room below: training less should hand back collateral damage without
# costing any erasure. This submits the generation for those checkpoints and
# chains the scoring behind it.
#
#   bash slurm/run_identity9_ck.sh              10 and 20, plus the rescore
#   CKPTS="000010 000015 000020" bash ...       other checkpoints
#   RESCORE=0 bash ...                          skip the ck30/50 rescore
#   DRY=1 bash ...                              print the sbatch lines only
#
# The rescore is here because the reference photographs changed from one per
# identity to five. Every number in results/identity9 predates that and is not
# comparable with anything produced from now on, so the old CSVs are moved
# aside rather than left to be picked up by --resume.
# =============================================================================

set -euo pipefail

REPO_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning
cd "$REPO_DIR"

CKPTS="${CKPTS:-000010 000020}"
TABLE="${TABLE:-configs/identity9_variants.tsv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos/identity9}"
PROMPTS_MODULE="${PROMPTS_MODULE:-identity_prompts}"
TAG="${TAG:-identity9}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
MASK_MODE="${MASK_MODE:-unclamped}"
IMAP_ALPHA="${IMAP_ALPHA:-7}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
RESCORE="${RESCORE:-1}"
DRY="${DRY:-0}"

# The command goes to stderr, because the callers capture stdout to read the
# job id back out of --parsable.
run() {
    printf '  %s\n' "$*" >&2
    if [ "$DRY" = "1" ]; then echo "  [dry run, not submitted]" >&2; return 0; fi
    "$@"
}

# --- what is on disk ---------------------------------------------------------
# The sweep directory is not worth hardcoding: find it from a variant that must
# exist, so renaming the sweep does not silently point this at nothing.
PROBE=$(ls -d outputs/*/donald_trump_demo_lr"${LEARNING_RATE//[.-]/}"/"$MASK_MODE"_alpha"$IMAP_ALPHA" 2>/dev/null | head -1 || true)
[ -n "$PROBE" ] || { echo "ERROR: no trained identity weights found under outputs/." >&2; exit 2; }
SWEEP_DIR=$(echo "$PROBE" | cut -d/ -f2)

N_VARIANTS=$(grep -c . "$TABLE")
N_PEOPLE=$(M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(len(importlib.import_module(os.environ["M"]).PEOPLE))')
N_TASKS=$((N_VARIANTS * N_PEOPLE))

echo "sweep dir : $SWEEP_DIR"
echo "table     : $TABLE  ($N_VARIANTS variants x $N_PEOPLE people = $N_TASKS tasks)"
echo "checkpoints: $CKPTS"
echo ""

# Every variant must carry the checkpoint, otherwise the generation would skip
# those columns quietly and the table would come out with holes that look like
# results.
MISSING=0
for CK in $CKPTS; do
    HAVE=$(ls -d outputs/"$SWEEP_DIR"/*/"$MASK_MODE"_alpha"$IMAP_ALPHA"/checkpoint-"$CK" 2>/dev/null | wc -l)
    printf "checkpoint-%s  %d/%d variants\n" "$CK" "$HAVE" "$N_VARIANTS"
    [ "$HAVE" -eq "$N_VARIANTS" ] || MISSING=1
done
if [ "$MISSING" = "1" ]; then
    echo ""
    echo "ERROR: a checkpoint is missing from some variants. Training wrote every"
    echo "CHECKPOINTING_STEPS steps, so a checkpoint that is not a multiple of"
    echo "that never existed -- retrain with a smaller CHECKPOINTING_STEPS"
    echo "before asking for it." >&2
    exit 2
fi
echo ""

# --- generation --------------------------------------------------------------
GEN_IDS=""
for CK in $CKPTS; do
    echo "generation for checkpoint-$CK:"
    JOB=$(run sbatch --parsable --array=0-$((N_TASKS - 1)) \
        --export="ALL,MODE=column,TABLE=$TABLE,VIDEOS_ROOT=$VIDEOS_ROOT,SWEEP_DIR=$SWEEP_DIR,PROMPTS_MODULE=$PROMPTS_MODULE,NUM_PROMPTS=$NUM_PROMPTS,MASK_MODE=$MASK_MODE,IMAP_ALPHA=$IMAP_ALPHA,LEARNING_RATE=$LEARNING_RATE,CHECKPOINT=checkpoint-$CK" \
        slurm/gen_celeb_gh200.sbatch)
    [ "$DRY" = "1" ] || { echo "    job $JOB"; GEN_IDS="${GEN_IDS:+$GEN_IDS:}$JOB"; }
done
echo ""

# --- scoring, behind the generation -----------------------------------------
# afterany, not afterok: one column failing should not cancel the scoring of
# the other twenty-six. --resume then fills whatever arrives late.
LR_TAG="${LEARNING_RATE//[.-]/}"
METHODS=""
for CK in $CKPTS; do
    for MAP in null person demo; do
        METHODS="${METHODS:+$METHODS }imap_${MAP}_lr${LR_TAG}_${CK}"
    done
done
N_METHODS=$(echo "$METHODS" | wc -w)

echo "scoring ($N_METHODS method labels):"
DEP=""
[ -n "$GEN_IDS" ] && DEP="--dependency=afterany:$GEN_IDS --kill-on-invalid-dep=yes"
# shellcheck disable=SC2086
SCORE=$(run sbatch --parsable $DEP --array=0-$((N_METHODS - 1)) \
    --export="ALL,VIDEOS_ROOT=$VIDEOS_ROOT,TAG=$TAG,PROMPTS_MODULE=$PROMPTS_MODULE,METHODS=$METHODS" \
    slurm/id_similarity_split_gh200.sbatch)
[ "$DRY" = "1" ] || echo "    job $SCORE  (waits for $GEN_IDS)"
echo ""

# --- rescore of what already exists -----------------------------------------
if [ "$RESCORE" = "1" ]; then
    echo "rescore of ck30/ck50 against the five-photograph references:"
    OLD="results/${TAG}_refs1"
    if compgen -G "results/$TAG/id_similarity_*.csv" > /dev/null; then
        run mkdir -p "$OLD"
        if [ "$DRY" != "1" ]; then
            mv results/"$TAG"/id_similarity_*.csv "$OLD"/
            echo "    old CSVs -> $OLD"
        fi
    fi
    RE=$(run sbatch --parsable slurm/id_similarity_split_gh200.sbatch)
    [ "$DRY" = "1" ] || echo "    job $RE"
    echo ""
fi

echo "squeue -u \$USER   to watch."
