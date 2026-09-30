#!/bin/bash
# =============================================================================
# ArcFace ID-Similarity over a generated identity tree.
#
#   sbatch --export=ALL,METHODS="imap_null_lr1e3_000030 imap_null_lr1e3_000050" \
#          benchmark/run_id_similarity.sbatch
#
# insightface is not part of the base environment. INSTALL=1 puts it and
# onnxruntime-gpu in before scoring; leave it unset once they are there.
#
# Output: results/<TAG>/id_similarity.csv
# =============================================================================

#SBATCH --job-name=id_sim
#SBATCH --output=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/id_sim_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plglukaszrudnik/repos/Video_Unlearning/logs/id_sim_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=72
#SBATCH --mem=120G

set -euo pipefail

BASE_DIR=/net/scratch/hscra/plgrid/plglukaszrudnik
REPO_DIR=$BASE_DIR/repos/Video_Unlearning
ENV_DIR=${ENV_DIR:-$BASE_DIR/conda/envs/eraser}

export HF_HOME=$REPO_DIR/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$REPO_DIR/.cache/torch
export TMPDIR=$REPO_DIR/.cache/tmp
# insightface downloads its model pack here on first use.
export INSIGHTFACE_HOME=$REPO_DIR/.cache/insightface
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
mkdir -p "$REPO_DIR/logs" "$TMPDIR" "$INSIGHTFACE_HOME"

module purge
module load CUDA/12.8.0
source "$BASE_DIR/conda/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"

: "${VIDEOS_ROOT:=videos/identity}"
: "${REFS_DIR:=data/identity_refs}"
: "${PROMPTS_MODULE:=identity_prompts}"
: "${TAG:=identity}"
: "${METHODS:?set METHODS to the method directory names to score}"
: "${FRAME_STRIDE:=1}"
# Grace has seventy-two cores and the job was asking for eight. There is no
# aarch64 GPU build of onnxruntime, so this runs on the CPU either way -- but
# on all of it, in parallel processes, rather than on one ninth of it.
: "${WORKERS:=0}"
: "${THREADS_PER_WORKER:=4}"
: "${DET_SIZE:=640}"
: "${INCLUDE_BASE:=1}"
: "${INSTALL:=}"
: "${RESUME:=1}"
: "${OUT:=results/$TAG/id_similarity.csv}"

if [ -n "$INSTALL" ]; then
    echo "Installing insightface and onnxruntime-gpu ..."
    pip install --no-input insightface onnxruntime-gpu || {
        echo "onnxruntime-gpu unavailable on this architecture; trying CPU build."
        pip install --no-input insightface onnxruntime
    }
fi

python3 -c "import insightface, onnxruntime; print('insightface', insightface.__version__);
print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"

# shellcheck disable=SC2206
METHOD_ARR=($METHODS)

CMD=(python3 benchmark/id_similarity.py
    --videos_root "$VIDEOS_ROOT"
    --refs_dir "$REFS_DIR"
    --prompts_module "$PROMPTS_MODULE"
    --methods "${METHOD_ARR[@]}"
    --det_size "$DET_SIZE"
    --frame_stride "$FRAME_STRIDE"
    --workers "$WORKERS"
    --threads_per_worker "$THREADS_PER_WORKER"
    --out "$OUT")

[ "$INCLUDE_BASE" = "1" ] && CMD+=(--include_base)
[ "$RESUME" = "1" ] && CMD+=(--resume)

printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
echo "----------------------------------------------------------"
exec "${CMD[@]}"
