#!/bin/bash
# =============================================================================
# Brand videos: thirty per brand from the base model, thirty from the erased
# one. Two arrays, nothing scored.
#
#   bash slurm/run_brands.sh                 base + generic ck50
#   CKPT=000030 bash slurm/run_brands.sh     a different checkpoint
#   MAPPING=rival TABLE=configs/brand_rival.tsv bash slurm/run_brands.sh
#   DRY=1 bash slurm/run_brands.sh           print the sbatch lines only
#
# The generator indexes columns as variant * n_brands + brand, because the
# identity benchmark needs every model asked for every identity -- that is
# where Preserve comes from. Brands here are not scored against each other, so
# only the diagonal is wanted: the model that erased Nike, asked for Nike.
# That is ten of the hundred cells, and the array is submitted as an explicit
# index list rather than a range.
#
# The diagonal only lands on the right cells if line i of the table is brand i
# of the prompts module, so the alignment is checked rather than assumed.
#
# Output: videos/brands/_shared/base/<brand>/   and
#         videos/brands/unlearn_<brand>/<brand>/
# =============================================================================

set -euo pipefail

REPO_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning
cd "$REPO_DIR"

CKPT="${CKPT:-000050}"
TABLE="${TABLE:-configs/brand_generic.tsv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-videos/brands}"
PROMPTS_MODULE="${PROMPTS_MODULE:-brand_prompts}"
SWEEP_DIR="${SWEEP_DIR:-brand_sweep}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
MASK_MODE="${MASK_MODE:-unclamped}"
IMAP_ALPHA="${IMAP_ALPHA:-7}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
METHOD_SUFFIX="${METHOD_SUFFIX:-}"
BASE_ONLY="${BASE_ONLY:-0}"
DRY="${DRY:-0}"

run() {
    printf '  %s\n' "$*" >&2
    if [ "$DRY" = "1" ]; then echo "  [dry run, not submitted]" >&2; return 0; fi
    "$@"
}

LR_TAG="${LEARNING_RATE//[.-]/}"

# Alignment check and diagonal in one pass. Anything wrong here would generate
# the right number of videos under the wrong labels, which is the failure that
# survives review, so it stops the script instead of warning.
DIAG=$(TABLE="$TABLE" M="$PROMPTS_MODULE" SWEEP_DIR="$SWEEP_DIR" \
       MASK_MODE="$MASK_MODE" IMAP_ALPHA="$IMAP_ALPHA" LR_TAG="$LR_TAG" \
       CKPT="$CKPT" python3 - <<'PY'
import importlib, os, sys
sys.path.insert(0, "benchmark")
from common import slugify

env = os.environ
brands = importlib.import_module(env["M"]).PEOPLE
rows = [l.rstrip("\n").split("\t") for l in open(env["TABLE"], encoding="utf-8") if l.strip()]

if len(rows) != len(brands):
    sys.exit("Table has %d rows, the prompts module has %d brands."
             % (len(rows), len(brands)))

missing = []
for i, (row, brand) in enumerate(zip(rows, brands)):
    if row[0] != brand:
        sys.exit("Line %d of the table is %r, but brand %d is %r. The diagonal "
                 "would generate the wrong cell; reorder the table."
                 % (i + 1, row[0], i, brand))
    path = os.path.join("outputs", env["SWEEP_DIR"],
                        "%s_%s_lr%s" % (slugify(row[0]), row[1], env["LR_TAG"]),
                        "%s_alpha%s" % (env["MASK_MODE"], env["IMAP_ALPHA"]),
                        "checkpoint-%s" % env["CKPT"])
    if not os.path.isdir(path):
        missing.append(path)

if missing:
    sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing))

n = len(brands)
print(",".join(str(i * n + i) for i in range(n)))
PY
)

N_BRANDS=$(M="$PROMPTS_MODULE" python3 -c 'import os,sys,importlib;sys.path.insert(0,"benchmark");print(len(importlib.import_module(os.environ["M"]).PEOPLE))')

echo "sweep dir  : $SWEEP_DIR"
echo "table      : $TABLE"
echo "prompts    : $PROMPTS_MODULE  ($N_BRANDS brands x $NUM_PROMPTS prompts)"
echo "checkpoint : $CKPT  (all $N_BRANDS present)"
echo "diagonal   : $DIAG"
echo "videos     : $VIDEOS_ROOT"
echo "total      : $((N_BRANDS * NUM_PROMPTS)) base + $((N_BRANDS * NUM_PROMPTS)) erased"
echo ""

COMMON="TABLE=$TABLE,VIDEOS_ROOT=$VIDEOS_ROOT,SWEEP_DIR=$SWEEP_DIR,PROMPTS_MODULE=$PROMPTS_MODULE,NUM_PROMPTS=$NUM_PROMPTS,MASK_MODE=$MASK_MODE,IMAP_ALPHA=$IMAP_ALPHA,LEARNING_RATE=$LEARNING_RATE,METHOD_SUFFIX=$METHOD_SUFFIX,LORA_WEIGHT=$LORA_WEIGHT,STRICT=1"

echo "baseline ($N_BRANDS tasks):"
BASE=$(run sbatch --parsable --array=0-$((N_BRANDS - 1)) \
    --export="ALL,MODE=base,$COMMON" slurm/gen_celeb_gh200.sbatch)
echo ""

if [ "$BASE_ONLY" = "1" ]; then
    [ "$DRY" = "1" ] || echo "baseline $BASE   (columns skipped, BASE_ONLY=1)"
    echo "squeue -u \$USER   to watch."
    exit 0
fi

echo "erased ($N_BRANDS tasks, diagonal):"
COLS=$(run sbatch --parsable --array="$DIAG" \
    --export="ALL,MODE=column,$COMMON,CHECKPOINT=checkpoint-$CKPT" \
    slurm/gen_celeb_gh200.sbatch)
echo ""

if [ "$DRY" != "1" ]; then
    echo "baseline $BASE   erased $COLS"
fi
echo "squeue -u \$USER   to watch."
