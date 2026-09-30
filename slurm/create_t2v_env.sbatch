#!/bin/bash
#SBATCH --job-name=t2v_env
#SBATCH --output=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_env_%j.out
#SBATCH --error=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_env_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --partition=dgxa100
#SBATCH --qos=quick

set -euo pipefail

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

export NVCC_PREPEND_FLAGS=""

export BASE_DIR="/shared/results/gmpolowl"
export REPO_DIR="$BASE_DIR/repozytorium/T2VUnlearning"
export ENV_DIR="$BASE_DIR/envs/t2v_unlearning"
export PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

export HF_DATASETS_CACHE="$BASE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/huggingface/hub"
export HF_XET_CACHE="$BASE_DIR/huggingface/xet"
export TORCH_HOME="$BASE_DIR/torch"
export HF_HOME="$BASE_DIR/huggingface"

export TMPDIR="$BASE_DIR/pip-tmp"
export PIP_CACHE_DIR="$BASE_DIR/pip-cache"
export CONDA_PKGS_DIRS="$BASE_DIR/conda-pkgs"
export PYTHONNOUSERSITE=1

mkdir -p "$REPO_DIR/logs"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_XET_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$CONDA_PKGS_DIRS"
mkdir -p "$(dirname "$ENV_DIR")"

eval "$(conda shell.bash hook)"

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.10 -y
fi

conda activate "$ENV_DIR"

python -m pip install --upgrade pip setuptools wheel

# Versions recommended by T2VUnlearning README.
python -m pip install \
    torch==2.6.0 \
    --index-url "$PYTORCH_CUDA_INDEX_URL"

python -m pip install \
    transformers==4.48.0 \
    accelerate==1.1.0

cd "$REPO_DIR/diffusers"
python -m pip install -e .

cd "$REPO_DIR/diffusers/examples/cogvideo"
python -m pip install -r requirements.txt
python -m pip install --no-deps --force-reinstall peft==0.14.0
python -m pip install opencv-python omegaconf imageio imageio-ffmpeg

python - <<'PY'
import torch
import transformers
import accelerate
import diffusers

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("diffusers:", diffusers.__version__)
PY

echo "Environment ready: $ENV_DIR"
