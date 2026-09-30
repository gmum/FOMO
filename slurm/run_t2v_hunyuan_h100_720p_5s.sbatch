#!/bin/bash
#SBATCH --job-name=t2v_hunyuan_720p
#SBATCH --output=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_hunyuan_720p_%j.out
#SBATCH --error=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_hunyuan_720p_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
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

PROMPT="${PROMPT:-A cinematic landscape at sunrise, slow camera movement, detailed clouds.}"
MODEL_PATH="${MODEL_PATH:-hunyuanvideo-community/HunyuanVideo}"
ERASER_PATH="${ERASER_PATH:-}"
LORA_PATH="${LORA_PATH:-}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/videos/hunyuan_1280x720_5s}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-75}"
FPS="${FPS:-15}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SEED="${SEED:-42}"

if [[ "$MODEL_PATH" == /* ]] && [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "ERROR: MODEL_PATH is local but does not contain model_index.json: $MODEL_PATH"
    echo "Point MODEL_PATH at a full HunyuanVideo diffusers directory or use the Hugging Face repo id: hunyuanvideo-community/HunyuanVideo"
    exit 1
fi

if [[ "$ERASER_PATH" == /* ]] && { [ ! -f "$ERASER_PATH/eraser_weights.pt" ] || [ ! -f "$ERASER_PATH/eraser_config.json" ]; }; then
    echo "ERROR: ERASER_PATH must contain eraser_weights.pt and eraser_config.json: $ERASER_PATH"
    exit 1
fi

if [ -n "$ERASER_PATH" ] && [ -n "$LORA_PATH" ]; then
    echo "ERROR: use either ERASER_PATH or LORA_PATH, not both."
    exit 1
fi

if [[ "$LORA_PATH" == /* ]] && [ ! -f "$LORA_PATH/pytorch_lora_weights.safetensors" ] && [ ! -f "$LORA_PATH/pytorch_lora_weights.bin" ]; then
    echo "ERROR: LORA_PATH must contain pytorch_lora_weights.safetensors or pytorch_lora_weights.bin: $LORA_PATH"
    exit 1
fi

if [ -z "$ERASER_PATH" ]; then
    if [ -z "$LORA_PATH" ]; then
        echo "ERASER_PATH and LORA_PATH are empty; running base HunyuanVideo."
    else
        echo "Using PEFT/diffusers LoRA: $LORA_PATH"
    fi
else
    echo "Using T2VUnlearning adapter: $ERASER_PATH"
fi

CMD=(python test_hunyuan.py
    --prompt="$PROMPT" \
    --model_path="$MODEL_PATH" \
    --eraser_rank=128 \
    --height="$HEIGHT" \
    --width="$WIDTH" \
    --num_frames="$NUM_FRAMES" \
    --fps="$FPS" \
    --num_inference_steps="$NUM_INFERENCE_STEPS" \
    --guidance_scale="$GUIDANCE_SCALE" \
    --output_path="$OUTPUT_PATH" \
    --seed="$SEED")

if [ -n "$ERASER_PATH" ]; then
    CMD+=(--eraser_path="$ERASER_PATH")
fi

if [ -n "$LORA_PATH" ]; then
    CMD+=(--lora_path="$LORA_PATH")
    CMD+=(--lora_weight="$LORA_WEIGHT")
fi

"${CMD[@]}"
