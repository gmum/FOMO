#!/bin/bash
# =============================================================================
# Generate the ESD vs ESD + L_V comparison, plus the untouched base model.
#
#   Index = concept * 3 + model
#     model 0  ->  base    no adapter
#     model 1  ->  esd     lamb_v = 0
#     model 2  ->  esd_v   lamb_v > 0
#
#   sbatch slurm/gen_esd_visual_gh200.sbatch                one row per concept
#   sbatch --array=0-2 slurm/gen_esd_visual_gh200.sbatch    dog only
#
# A model without trained weights exits quietly, so the array can hang off the
# training job as a dependency and be re-run later to fill the gaps.
#
# Output: videos/esd_visual/<concept>/<base|esd|esd_v>/NN.mp4
# =============================================================================

#SBATCH --job-name=esd_vis
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_vis_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_vis_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --array=0-11

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
ENV_DIR=$BASE_DIR/conda/envs/eraser

export HF_HOME=$REPO_DIR/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$REPO_DIR/.cache/torch
export TMPDIR=$REPO_DIR/.cache/tmp
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$REPO_DIR/logs" "$TMPDIR"

module purge
module load CUDA/12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

: "${PROMPTS_MODULE:=visual_prompts}"
: "${VIDEOS_ROOT:=videos/esd_visual}"
: "${SWEEP_DIR:=esd_visual}"
: "${LEARNING_RATE:=2e-5}"
: "${CHECKPOINT:=}"
LR_TAG=$(echo "$LEARNING_RATE" | tr -d '.-')

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CON_IDX=$((IDX / 3))
MOD_IDX=$((IDX % 3))

CONCEPT=$(I="$CON_IDX" M="$PROMPTS_MODULE" python3 -c '
import os, sys, importlib
sys.path.insert(0, "benchmark")
subjects = importlib.import_module(os.environ["M"]).PEOPLE
i = int(os.environ["I"])
if i >= len(subjects):
    sys.exit("ERROR: concept index %d out of range 0..%d" % (i, len(subjects) - 1))
print(subjects[i])
')
SLUG=$(C="$CONCEPT" python3 -c 'import os,sys;sys.path.insert(0,"benchmark");from common import slugify;print(slugify(os.environ["C"]))')

case "$MOD_IDX" in
    0) METHOD=base;  VARIANT="" ;;
    1) METHOD=esd;   VARIANT=esd ;;
    2) METHOD=esd_v; VARIANT=esd_v ;;
esac

echo "Task $IDX | concept '$CONCEPT' | method $METHOD"

if [ -z "$VARIANT" ]; then
    exec python3 benchmark/generate_celeb.py \
        --person "$CONCEPT" \
        --output_dir "$VIDEOS_ROOT/$SLUG" \
        --method base \
        --prompts_module "$PROMPTS_MODULE"
fi

WEIGHTS="$REPO_DIR/outputs/${SWEEP_DIR}/${SLUG}_${VARIANT}_lr${LR_TAG}"
[ -n "$CHECKPOINT" ] && WEIGHTS="$WEIGHTS/$CHECKPOINT"
echo "  weights: $WEIGHTS"

if ! compgen -G "$WEIGHTS/checkpoint-*" > /dev/null 2>&1 \
   && [ ! -f "$WEIGHTS/pytorch_lora_weights.safetensors" ]; then
    echo "SKIP: no trained weights under $WEIGHTS yet."
    exit 0
fi

exec python3 benchmark/generate_celeb.py \
    --person "$CONCEPT" \
    --output_dir "$VIDEOS_ROOT/$SLUG" \
    --method "$METHOD" \
    --lora "$WEIGHTS" \
    --prompts_module "$PROMPTS_MODULE"
