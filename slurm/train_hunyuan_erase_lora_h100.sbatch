#!/bin/bash
#SBATCH --job-name=t2v_erase_lora
#SBATCH --output=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_erase_lora_%j.out
#SBATCH --error=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_erase_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G
#SBATCH --partition=dgxh100
#SBATCH --qos=quick

set -euo pipefail

export BASE_DIR="/shared/results/gmpolowl"
export REPO_DIR="$BASE_DIR/repozytorium/T2VUnlearning"
export ENV_DIR="$BASE_DIR/envs/t2v_unlearning"

export HF_DATASETS_CACHE="$BASE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/huggingface/hub"
export HF_XET_CACHE="$BASE_DIR/huggingface/xet"
export TORCH_HOME="$BASE_DIR/torch"
export HF_HOME="$BASE_DIR/huggingface"
export TMPDIR="$BASE_DIR/pip-tmp"
export PIP_CACHE_DIR="$BASE_DIR/pip-cache"
export CONDA_PKGS_DIRS="$BASE_DIR/conda-pkgs"
export PYTHONNOUSERSITE=1

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/outputs" "$REPO_DIR/data"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

cd "$REPO_DIR"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
python - <<'PY'
import torch
import diffusers
import peft
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, torch.cuda.is_available())
print("diffusers:", diffusers.__version__, diffusers.__file__)
print("peft:", peft.__version__, peft.__file__)
PY

CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/hunyuan_erase_lora_cat.yaml}"
CONCEPT="${CONCEPT:-cat}"
NEUTRAL_PROMPT_OVERRIDE="${NEUTRAL_PROMPT_OVERRIDE:-}"
USE_EMPTY_NEUTRAL_PROMPT="${USE_EMPTY_NEUTRAL_PROMPT:-}"
MODEL_PATH="${MODEL_PATH:-hunyuanvideo-community/HunyuanVideo}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data/$CONCEPT}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/hunyuan_lora_$CONCEPT}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-}"
RANK="${RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
ADAM_BETA1="${ADAM_BETA1:-}"
ADAM_BETA2="${ADAM_BETA2:-}"
ADAM_EPSILON="${ADAM_EPSILON:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
LAMB_ESD="${LAMB_ESD:-}"
LAMB_ATTN="${LAMB_ATTN:-}"
ESD_FULL_VIDEO_WEIGHT="${ESD_FULL_VIDEO_WEIGHT:-}"
ESD_FIRST_FRAME_WEIGHT="${ESD_FIRST_FRAME_WEIGHT:-}"
NEGATIVE_GUIDANCE="${NEGATIVE_GUIDANCE:-}"
TEACHER_MODE="${TEACHER_MODE:-}"
ATTN_BLOCKS="${ATTN_BLOCKS:-}"
TARGET_MODULES="${TARGET_MODULES:-}"
TRAIN_WIDTH="${TRAIN_WIDTH:-}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-}"
TRAIN_NUM_FRAMES="${TRAIN_NUM_FRAMES:-}"
NUM_BOOTSTRAP_VIDEOS="${NUM_BOOTSTRAP_VIDEOS:-}"
BOOTSTRAP_WIDTH="${BOOTSTRAP_WIDTH:-$TRAIN_WIDTH}"
BOOTSTRAP_HEIGHT="${BOOTSTRAP_HEIGHT:-$TRAIN_HEIGHT}"
BOOTSTRAP_NUM_FRAMES="${BOOTSTRAP_NUM_FRAMES:-$TRAIN_NUM_FRAMES}"
BOOTSTRAP_PROMPT_OVERRIDE="${BOOTSTRAP_PROMPT_OVERRIDE:-}"
BOOTSTRAP_NEUTRAL_PROMPT_OVERRIDE="${BOOTSTRAP_NEUTRAL_PROMPT_OVERRIDE:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"
PROMPT_ONLY_ESD="${PROMPT_ONLY_ESD:-}"

if [[ "$MODEL_PATH" == /* ]] && [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "ERROR: MODEL_PATH is local but does not contain model_index.json: $MODEL_PATH"
    echo "Point MODEL_PATH at a full HunyuanVideo diffusers directory or use hunyuanvideo-community/HunyuanVideo."
    exit 1
fi

CMD=(python train_hunyuan_erase_lora.py
    --config "$CONFIG_PATH" \
    --concept "$CONCEPT" \
    --model_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR")

if [ -n "$MAX_TRAIN_STEPS" ]; then
    CMD+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

if [ -n "$CHECKPOINTING_STEPS" ]; then
    CMD+=(--checkpointing_steps "$CHECKPOINTING_STEPS")
fi

if [ -n "$RANK" ]; then
    CMD+=(--rank "$RANK")
fi

if [ -n "$LORA_ALPHA" ]; then
    CMD+=(--lora_alpha "$LORA_ALPHA")
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD+=(--learning_rate "$LEARNING_RATE")
fi

if [ -n "$NEUTRAL_PROMPT_OVERRIDE" ]; then
    CMD+=(--neutral_prompt_override "$NEUTRAL_PROMPT_OVERRIDE")
fi

if [ -n "$USE_EMPTY_NEUTRAL_PROMPT" ]; then
    CMD+=(--use_empty_neutral_prompt)
fi

if [ -n "$ADAM_BETA1" ]; then
    CMD+=(--adam_beta1 "$ADAM_BETA1")
fi

if [ -n "$ADAM_BETA2" ]; then
    CMD+=(--adam_beta2 "$ADAM_BETA2")
fi

if [ -n "$ADAM_EPSILON" ]; then
    CMD+=(--adam_epsilon "$ADAM_EPSILON")
fi

if [ -n "$WEIGHT_DECAY" ]; then
    CMD+=(--weight_decay "$WEIGHT_DECAY")
fi

if [ -n "$LAMB_ESD" ]; then
    CMD+=(--lamb_esd "$LAMB_ESD")
fi

if [ -n "$LAMB_ATTN" ]; then
    CMD+=(--lamb_attn "$LAMB_ATTN")
fi

if [ -n "$ESD_FULL_VIDEO_WEIGHT" ]; then
    CMD+=(--esd_full_video_weight "$ESD_FULL_VIDEO_WEIGHT")
fi

if [ -n "$ESD_FIRST_FRAME_WEIGHT" ]; then
    CMD+=(--esd_first_frame_weight "$ESD_FIRST_FRAME_WEIGHT")
fi

if [ -n "$NEGATIVE_GUIDANCE" ]; then
    CMD+=(--negative_guidance "$NEGATIVE_GUIDANCE")
fi

if [ -n "$TEACHER_MODE" ]; then
    CMD+=(--teacher_mode "$TEACHER_MODE")
fi

if [ -n "$ATTN_BLOCKS" ]; then
    CMD+=(--attn_blocks "$ATTN_BLOCKS")
fi

if [ -n "$TARGET_MODULES" ]; then
    CMD+=(--target_modules "$TARGET_MODULES")
fi

if [ -n "$TRAIN_WIDTH" ]; then
    CMD+=(--train_width "$TRAIN_WIDTH")
fi

if [ -n "$TRAIN_HEIGHT" ]; then
    CMD+=(--train_height "$TRAIN_HEIGHT")
fi

if [ -n "$TRAIN_NUM_FRAMES" ]; then
    CMD+=(--train_num_frames "$TRAIN_NUM_FRAMES")
fi

if [ -n "$NUM_BOOTSTRAP_VIDEOS" ]; then
    CMD+=(--num_bootstrap_videos "$NUM_BOOTSTRAP_VIDEOS")
fi

if [ -n "$BOOTSTRAP_WIDTH" ]; then
    CMD+=(--bootstrap_width "$BOOTSTRAP_WIDTH")
fi

if [ -n "$BOOTSTRAP_HEIGHT" ]; then
    CMD+=(--bootstrap_height "$BOOTSTRAP_HEIGHT")
fi

if [ -n "$BOOTSTRAP_NUM_FRAMES" ]; then
    CMD+=(--bootstrap_num_frames "$BOOTSTRAP_NUM_FRAMES")
fi

if [ -n "$BOOTSTRAP_PROMPT_OVERRIDE" ]; then
    CMD+=(--bootstrap_prompt_override "$BOOTSTRAP_PROMPT_OVERRIDE")
fi

if [ -n "$BOOTSTRAP_NEUTRAL_PROMPT_OVERRIDE" ]; then
    CMD+=(--bootstrap_neutral_prompt_override "$BOOTSTRAP_NEUTRAL_PROMPT_OVERRIDE")
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "TRUE" ]]; then
    CMD+=(--gradient_checkpointing)
fi

if [[ "$PROMPT_ONLY_ESD" == "1" || "$PROMPT_ONLY_ESD" == "true" || "$PROMPT_ONLY_ESD" == "TRUE" ]]; then
    CMD+=(--prompt_only_esd)
fi

"${CMD[@]}"

echo "Training finished. LoRA output: $OUTPUT_DIR"
