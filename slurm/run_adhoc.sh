#!/bin/bash
# =============================================================================
# Submit the hand-written prompt list, sized to however many prompts it has.
#
#   bash slurm/run_adhoc.sh
#   TAG=skulls bash slurm/run_adhoc.sh
#   NUM_FRAMES=49 ADHOC_CHUNK=10 TAG=long bash slurm/run_adhoc.sh
#   LORA=outputs/swap_visual_sweep/skull_knight_helmet_lr1e3/unclamped_alpha7/checkpoint-000050 \
#     TAG=skull_ck50 bash slurm/run_adhoc.sh
#   DRY=1 bash slurm/run_adhoc.sh
#
# Every slice writes into videos/adhoc/<TAG>/ and numbers its output by line
# number, so NNN.mp4 is line NNN of the file regardless of which job made it.
# =============================================================================

set -euo pipefail

REPO_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning
cd "$REPO_DIR"

: "${ADHOC_FILE:=configs/adhoc_prompts.txt}"
: "${ADHOC_CHUNK:=10}"
: "${TAG:=base}"
: "${LORA:=}"
: "${LORA_WEIGHT:=1.0}"
: "${SEED:=42}"
: "${NUM_FRAMES:=17}"
: "${DRY:=0}"

N_PROMPTS=$(grep -cve '^\s*$' -e '^\s*#' "$ADHOC_FILE" || true)
[ "${N_PROMPTS:-0}" -gt 0 ] || { echo "ERROR: no prompts in $ADHOC_FILE" >&2; exit 2; }
N_PARTS=$(( (N_PROMPTS + ADHOC_CHUNK - 1) / ADHOC_CHUNK ))

echo "list    : $ADHOC_FILE"
echo "prompts : $N_PROMPTS in $N_PARTS slice(s) of $ADHOC_CHUNK"
echo "frames  : $NUM_FRAMES     seeds: $SEED .. $(( SEED + N_PROMPTS - 1 ))"
echo "tag     : $TAG"
[ -n "$LORA" ] && echo "lora    : $LORA (weight $LORA_WEIGHT)"
echo "output  : videos/adhoc/$TAG/NNN.mp4"
echo ""

CMD=(sbatch --array=0-$((N_PARTS - 1))
     --export="ALL,ADHOC_FILE=$ADHOC_FILE,ADHOC_CHUNK=$ADHOC_CHUNK,TAG=$TAG,LORA=$LORA,LORA_WEIGHT=$LORA_WEIGHT,SEED=$SEED,NUM_FRAMES=$NUM_FRAMES"
     slurm/gen_adhoc_gh200.sbatch)

printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
if [ "$DRY" = "1" ]; then
    echo "  [dry run, not submitted]"
    exit 0
fi
"${CMD[@]}"
