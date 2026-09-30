#!/bin/bash
# =============================================================================
# The operating-point grid: two extrapolation strengths by three checkpoints.
#
# Two knobs move along the same erasure/damage trade-off and neither has been
# measured against the other. Alpha at 3 instead of 7 raised Preserve from
# 0.1993 to 0.3229 on an earlier run, at the cost of Erase going 0.0301 ->
# 0.0649 -- still a fifth of the recognition threshold. Stopping at ten steps
# instead of thirty raised it again at no cost at all. Whether the two gains
# add or reach for the same slack is the question this answers.
#
# Six configurations, each one call to run_identity9_face.sh, which submits the
# baseline (already generated, so it skips), the columns, and a scoring job
# that waits on them.
#
#   bash slurm/run_identity9_grid.sh
#   NUM_PROMPTS=10 bash ...        a cheaper pass that only ranks the cells
#   ALPHAS="3" CKPTS="000005" ...  one cell
#   DRY=1 bash ...                 print without submitting
#
# The full thirty prompts by default. Ten would be enough to rank the cells,
# but then the winner has to be regenerated before it can be reported, and the
# Original row -- which comes from the thirty-prompt baseline -- would be
# averaged over a different sample than the cells it normalises.
# =============================================================================

set -euo pipefail

REPO_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning
cd "$REPO_DIR"

ALPHAS="${ALPHAS:-7 3}"
CKPTS="${CKPTS:-000003 000005 000010}"
SWEEP_DIR="${SWEEP_DIR:-identity9_early}"
TABLE="${TABLE:-configs/identity9_demo.tsv}"
PROMPTS_MODULE="${PROMPTS_MODULE:-identity_prompts_face}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos/identity9_face}"
TAG="${TAG:-identity9_face}"
# Training job ids, colon separated, when the weights are still being made:
#   AFTER=22956610:22956612 bash slurm/run_identity9_grid.sh
AFTER="${AFTER:-}"
DRY="${DRY:-0}"

echo "sweep dir : $SWEEP_DIR"
echo "grid      : alpha {$ALPHAS} x checkpoint {$CKPTS}"
echo "prompts   : $NUM_PROMPTS"
echo ""

for A in $ALPHAS; do
    for CK in $CKPTS; do
        echo "=============== alpha $A, checkpoint $CK ==============="
        SWEEP_DIR="$SWEEP_DIR" TABLE="$TABLE" MAPPINGS=demo \
        IMAP_ALPHA="$A" METHOD_SUFFIX="_a$A" CKPT="$CK" \
        NUM_PROMPTS="$NUM_PROMPTS" VIDEOS_ROOT="$VIDEOS_ROOT" TAG="$TAG" \
        PROMPTS_MODULE="$PROMPTS_MODULE" AFTER="$AFTER" DRY="$DRY" \
        bash slurm/run_identity9_face.sh
        echo ""
    done
done

echo "Method labels to expect in results/$TAG:"
for A in $ALPHAS; do
    for CK in $CKPTS; do
        echo "  imap_demo_lr1e3_a${A}_${CK}"
    done
done
