#!/bin/bash
# =============================================================================
# ESD on "dog", reproducing the run that worked in August.
#
# Recovered from commit d9fa6e2 (train_va.sbatch). The settings that matter and
# that the later Imagenette sweep got wrong:
#
#   learning_rate 0.003     the sweep used 2e-5, a hundred and fifty times less,
#                           and even its 1e-3 arm stayed below this
#   max_train_steps 1000    the sweep used 250
#   weight_decay 1e-5       the sweep used 0.01
#   negative_guidance 1.0   the sweep used 2.0
#   real bootstrap videos   the sweep trained prompt-only
#   768x432, 17 frames      33 frames is in the config and was overridden on the
#                           command line, because L_V holds full activations
#
#   sbatch --export=ALL,LAMB_V=0.0 slurm/train_esd_dog_repro.sbatch   plain ESD
#   sbatch --export=ALL,LAMB_V=0.5 slurm/train_esd_dog_repro.sbatch   ESD + L_V
#
# Output: outputs/esd_repro/dog_lamb<LAMB_V>/checkpoint-*
# =============================================================================

#SBATCH --job-name=esd_repro
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_repro_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/esd_repro_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/outputs" "$TMPDIR"

module purge
module load CUDA/12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

# The v1 config is the v_steps=1 variant; hunyuan_erase_lora_dog_v.yaml is the
# v_steps=3 one. Both carry the learning rate and the optimiser settings that
# made this work.
: "${CONFIG:=configs/hunyuan_erase_lora_dog_v1.yaml}"
: "${CONCEPT:=dog}"
: "${LAMB_V:=0.5}"
: "${V_STEPS:=1}"
: "${V_BLOCK_MIN:=20}"
: "${V_EARLY_PROB:=0.5}"
: "${DATASET_DIR:=$REPO_DIR/data/dog_alone_768x432_20_gc}"
: "${TRAIN_NUM_FRAMES:=17}"
: "${MAX_TRAIN_STEPS:=1000}"
: "${CHECKPOINTING_STEPS:=50}"
: "${SWEEP_DIR:=esd_repro}"

TAG=$(echo "$LAMB_V" | tr -d '.')
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/${SWEEP_DIR}/${CONCEPT}_lamb${TAG}}"
mkdir -p "$OUTPUT_DIR" "$DATASET_DIR"

# The L_V knobs are substituted into a copy of the config rather than passed as
# flags, which is how the original ran. The flags exist now and would work too,
# but a generated config leaves a record of exactly what this job used.
GEN_CONFIG="$OUTPUT_DIR/config.yaml"
sed -e "s|^lamb_v:.*|lamb_v: $LAMB_V|" \
    -e "s|^v_steps:.*|v_steps: $V_STEPS|" \
    -e "s|^v_block_min:.*|v_block_min: $V_BLOCK_MIN|" \
    -e "s|^v_early_prob:.*|v_early_prob: $V_EARLY_PROB|" \
    "$CONFIG" > "$GEN_CONFIG"

echo "Host: $(hostname)  $(date)"
echo "Concept : $CONCEPT"
echo "Config  : $CONFIG -> $GEN_CONFIG"
echo "Dataset : $DATASET_DIR"
echo "Output  : $OUTPUT_DIR"
grep -E "^(learning_rate|weight_decay|negative_guidance|rank|lora_alpha|prompt_only_esd|lamb_v|v_steps|v_disable_gc):" "$GEN_CONFIG"
echo "----------------------------------------------------------"

exec python train_hunyuan_erase_lora.py \
    --config "$GEN_CONFIG" \
    --concept "$CONCEPT" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_num_frames "$TRAIN_NUM_FRAMES" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --checkpointing_steps "$CHECKPOINTING_STEPS" \
    --gradient_checkpointing
