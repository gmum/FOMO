#!/bin/bash
#SBATCH --job-name=hunyuan_vbank
#SBATCH --output=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/hunyuan_vbank_%j.out
#SBATCH --error=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/hunyuan_vbank_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
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

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/videos"
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

PROMPT="${PROMPT:-a dog sitting on the bed}"
MODEL_PATH="${MODEL_PATH:-hunyuanvideo-community/HunyuanVideo}"
LORA_PATH="${LORA_PATH:?set LORA_PATH to a Hunyuan LoRA checkpoint or run directory}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/videos/vbank_dog_sitting_bed_768x432_seed42}"
HEIGHT="${HEIGHT:-432}"
WIDTH="${WIDTH:-768}"
NUM_FRAMES="${NUM_FRAMES:-75}"
FPS="${FPS:-15}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SEED="${SEED:-42}"
INJECT_STEPS="${INJECT_STEPS:-5}"
SWEEP_INJECT_STEPS="${SWEEP_INJECT_STEPS:-0}"
BLOCK_MIN="${BLOCK_MIN:-20}"
DTYPE="${DTYPE:-bfloat16}"

if [[ "$MODEL_PATH" == /* ]] && [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "ERROR: MODEL_PATH is local but does not contain model_index.json: $MODEL_PATH"
    echo "Use a full HunyuanVideo diffusers directory or: hunyuanvideo-community/HunyuanVideo"
    exit 1
fi

if [[ "$LORA_PATH" == /* ]] && [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LORA_PATH does not exist or is not a directory: $LORA_PATH"
    exit 1
fi

echo "Prompt: $PROMPT"
echo "Model: $MODEL_PATH"
echo "LoRA: $LORA_PATH"
echo "LoRA weight: $LORA_WEIGHT"
echo "Resolution/frames: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames @ ${FPS} fps"
echo "Steps/guidance/seed: ${NUM_INFERENCE_STEPS}, ${GUIDANCE_SCALE}, ${SEED}"
echo "V injection: first ${INJECT_STEPS} steps, single block index > ${BLOCK_MIN}"
echo "Sweep injection steps: ${SWEEP_INJECT_STEPS}"

CMD=(python test_hunyuan_vbank.py
    --prompt="$PROMPT" \
    --model_path="$MODEL_PATH" \
    --lora_path="$LORA_PATH" \
    --lora_weight="$LORA_WEIGHT" \
    --height="$HEIGHT" \
    --width="$WIDTH" \
    --num_frames="$NUM_FRAMES" \
    --fps="$FPS" \
    --num_inference_steps="$NUM_INFERENCE_STEPS" \
    --guidance_scale="$GUIDANCE_SCALE" \
    --output_path="$OUTPUT_PATH" \
    --seed="$SEED" \
    --inject_steps="$INJECT_STEPS" \
    --block_min="$BLOCK_MIN" \
    --dtype="$DTYPE")

if [ "$SWEEP_INJECT_STEPS" = "1" ]; then
    CMD+=(--sweep_inject_steps)
fi

"${CMD[@]}"
