#!/bin/bash
#SBATCH --job-name=t2v_hunyuan
#SBATCH --output=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_hunyuan_%j.out
#SBATCH --error=/shared/results/gmpolowl/repozytorium/T2VUnlearning/logs/t2v_hunyuan_%j.err
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
mkdir -p "$REPO_DIR/videos"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

cd "$REPO_DIR"

python test_hunyuan.py \
    --prompt="A cinematic landscape at sunrise, slow camera movement, detailed clouds." \
    --model_path="$BASE_DIR/models/hunyuanvideo" \
    --eraser_path="$BASE_DIR/models/t2v_unlearning/hunyuan_adapter" \
    --eraser_rank=128 \
    --num_frames=9 \
    --generate_clean \
    --output_path="$REPO_DIR/videos/hunyuan_test" \
    --seed=42
