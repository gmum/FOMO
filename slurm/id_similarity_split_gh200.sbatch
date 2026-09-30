#!/bin/bash
# =============================================================================
# ID-Similarity over videos/identity9, one array task per method label.
#
# Split rather than one long job for a reason that cost two hours once: a
# single unreadable mp4 took down a run that was three quarters finished, and
# the CSV was only written at the end. Now a failure costs one sixth, --resume
# picks up where the file left off, and every shard carries its own Original
# row so make_id_table.py can read it without any merging.
#
#   sbatch slurm/id_similarity_split_gh200.sbatch
#
# Output: results/identity9/id_similarity_<method>.csv and _per_video.csv
# =============================================================================

#SBATCH --job-name=id_split
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/id_split_%A_%a.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/id_split_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --array=0-5

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
# The environment with the locally built onnxruntime that has a CUDA provider.
# There is no aarch64 GPU wheel, so this one was compiled on a GH200 node; the
# stock `eraser` environment still has the CPU build and is left alone.
ENV_DIR=${ENV_DIR:-$BASE_DIR/conda/envs/eraser_ort}

module purge
module load CUDA/12.8.0
module load cuDNN/9.2.1.18-CUDA-12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

export INSIGHTFACE_HOME=$REPO_DIR/.cache/insightface
export HF_HOME=$REPO_DIR/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
mkdir -p "$REPO_DIR/results/identity9" "$REPO_DIR/logs"

: "${VIDEOS_ROOT:=videos/identity9}"
: "${REFS_DIR:=data/identity_refs}"
: "${PROMPTS_MODULE:=identity_prompts}"
: "${TAG:=identity9}"
: "${DET_SIZE:=640}"
: "${FRAME_STRIDE:=1}"

# Space-separated METHODS overrides the list, so the same script scores any
# tree. Remember to match --array to the number of labels: a task index past
# the end exits quietly rather than scoring the wrong thing.
#
#   sbatch --array=0-15 --export=ALL,VIDEOS_ROOT=videos/identity,TAG=identity,\
#   METHODS="imap_null_lr1e3_a7_000030 imap_null_lr1e3_a3_000030 ..." \
#     slurm/id_similarity_split_gh200.sbatch
: "${METHODS:=imap_null_lr1e3_000030 imap_null_lr1e3_000050 imap_person_lr1e3_000030 imap_person_lr1e3_000050 imap_demo_lr1e3_000030 imap_demo_lr1e3_000050}"
# shellcheck disable=SC2206
METHODS=($METHODS)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#METHODS[@]}" ]; then
    echo "Nothing at index $IDX; the array has ${#METHODS[@]} methods."
    exit 0
fi
M="${METHODS[$IDX]}"

echo "Host: $(hostname)  $(date)"
echo "Task $IDX -> $M"
echo "----------------------------------------------------------"

# One worker on purpose: the detector and ArcFace both run on the single GPU,
# so extra processes would only queue up behind each other on it.
exec python3 benchmark/id_similarity.py \
    --videos_root "$VIDEOS_ROOT" \
    --refs_dir "$REFS_DIR" \
    --prompts_module "$PROMPTS_MODULE" \
    --include_base --resume \
    --workers 1 \
    --det_size "$DET_SIZE" \
    --frame_stride "$FRAME_STRIDE" \
    --flush_every 10 \
    --methods "$M" \
    --out "results/$TAG/id_similarity_$M.csv"
