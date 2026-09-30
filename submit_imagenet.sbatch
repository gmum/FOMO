#!/bin/bash
# =============================================================================
# Erasure training for all 10 ImageNet classes of Table 1, as a job array.
#
# One array task = one class. The protocol erases a single concept at a time
# and measures preservation on the remaining nine, so every column of the
# table needs its own training run.
#
# All ten classes, at most 3 running at once:
#   sbatch --array=0-9%3 submit_imagenet.sbatch
#
# A single class (indices listed below):
#   sbatch --array=5 submit_imagenet.sbatch
#
# Retrain even if final weights already exist:
#   sbatch --array=0-9%3 --export=ALL,FORCE=1 submit_imagenet.sbatch
#
# Different hyperparameters for the whole sweep:
#   sbatch --array=0-9%3 --export=ALL,LAMB_V=0.1,MAX_TRAIN_STEPS=400 submit_imagenet.sbatch
#
# Class indices:
#   0 cassette player   5 garbage truck
#   1 chain saw         6 English springer
#   2 church            7 golf ball
#   3 gas pump          8 parachute
#   4 tench             9 French horn
# =============================================================================

#SBATCH --job-name=t2v_in_arr
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/t2v_in_arr_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/t2v_in_arr_%A_%a.err

set -euo pipefail

# Order must stay stable — the array index points into this list.
CONCEPTS=(
    "cassette player"   # 0
    "chain saw"         # 1
    "church"            # 2
    "gas pump"          # 3
    "tench"             # 4
    "garbage truck"     # 5
    "English springer"  # 6
    "golf ball"         # 7
    "parachute"         # 8
    "French horn"       # 9
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#CONCEPTS[@]}" ]; then
    echo "ERROR: index $IDX out of range 0..$(( ${#CONCEPTS[@]} - 1 ))"
    exit 1
fi
CONCEPT="${CONCEPTS[$IDX]}"
SLUG="$(echo "$CONCEPT" | tr ' ' '_' | tr -cd '[:alnum:]_')"

export BASE_DIR="/net/scratch/hscra/plgrid/plglukaszrudnik"
export REPO_DIR="$BASE_DIR/repos/Video_Unlearning"
export CONDA_SH="$BASE_DIR/conda/etc/profile.d/conda.sh"
export ENV_NAME="${ENV_NAME:-eraser}"

export HF_HOME="$BASE_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$BASE_DIR/cache/torch"
export TMPDIR="$BASE_DIR/cache/tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/outputs" "$TMPDIR"

echo "Task $IDX -> '$CONCEPT' | $(hostname) | $(date)"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO_DIR"

# --- hyperparameters ---------------------------------------------------------
LAMB_V="${LAMB_V:-0.5}"
V_STEPS="${V_STEPS:-1}"
V_BLOCK_MIN="${V_BLOCK_MIN:-20}"
V_EARLY_PROB="${V_EARLY_PROB:-0.5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50}"
NUM_BOOTSTRAP_VIDEOS="${NUM_BOOTSTRAP_VIDEOS:-10}"
TRAIN_NUM_FRAMES="${TRAIN_NUM_FRAMES:-17}"

DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data/in_${SLUG}_768x432}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/in_${SLUG}_lamb${LAMB_V/./}_v${V_STEPS}}"

# Skip only when the FINAL weights exist. A run interrupted halfway leaves
# checkpoint-* directories but no top-level file, and should restart.
if [ -z "${FORCE:-}" ] && [ -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]; then
    echo "[skip] '$CONCEPT' already trained -> $OUTPUT_DIR"
    echo "       pass FORCE=1 to retrain."
    exit 0
fi

# --- config generated from the base YAML ------------------------------------
BASE_CONFIG="${BASE_CONFIG:-$REPO_DIR/configs/hunyuan_erase_lora_dog_v.yaml}"
GEN_CONFIG="$REPO_DIR/configs/generated_in_${SLUG}.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: base config not found: $BASE_CONFIG"
    exit 1
fi

sed -e "s|^lamb_v:.*|lamb_v: $LAMB_V|" \
    -e "s|^v_steps:.*|v_steps: $V_STEPS|" \
    -e "s|^v_block_min:.*|v_block_min: $V_BLOCK_MIN|" \
    -e "s|^v_early_prob:.*|v_early_prob: $V_EARLY_PROB|" \
    "$BASE_CONFIG" > "$GEN_CONFIG"

# Neutral bootstrap prompt. The dog-specific one ("full body ... no owner,
# no hands") makes no sense for objects such as church or gas pump.
if [ -z "${BOOTSTRAP_PROMPT_OVERRIDE:-}" ]; then
    BOOTSTRAP_PROMPT_OVERRIDE='a realistic video of a {concept}'
fi

echo "config     : $GEN_CONFIG"
grep -E "^(lamb_v|v_steps|v_block_min|v_early_prob):" "$GEN_CONFIG"
echo "dataset    : $DATASET_DIR"
echo "output     : $OUTPUT_DIR"
echo "bootstrap  : $BOOTSTRAP_PROMPT_OVERRIDE"
echo "----------------------------------------------------------"

python train_hunyuan_erase_lora.py \
    --config "$GEN_CONFIG" \
    --concept "$CONCEPT" \
    --model_path "${MODEL_PATH:-hunyuanvideo-community/HunyuanVideo}" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --bootstrap_prompt_override "$BOOTSTRAP_PROMPT_OVERRIDE" \
    --num_bootstrap_videos "$NUM_BOOTSTRAP_VIDEOS" \
    --train_num_frames "$TRAIN_NUM_FRAMES" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --checkpointing_steps "$CHECKPOINTING_STEPS" \
    --gradient_checkpointing

echo "Task $IDX ('$CONCEPT') done -> $OUTPUT_DIR | $(date)"