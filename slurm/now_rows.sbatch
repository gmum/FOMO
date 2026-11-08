#!/bin/bash
# =============================================================================
# Train and generate several rows in a single job, for a QoS that takes no job
# arrays. PLGrid's `now` is one: it counts array tasks one by one against
# MaxSubmitPU, so an array of a hundred is refused outright while one long job
# starts within minutes.
#
#   sbatch --qos=now --export=ALL,CONFIGS="experiments/cogvideox_mapping1.yaml \
#          experiments/cogvideox_null.yaml" slurm/now_rows.sbatch
#
# Resumable, and meant to be resubmitted unchanged until it prints `done`:
# generation skips videos that exist and training skips a concept whose pinned
# checkpoint is on disk, so a job cut off at the wall clock loses nothing but
# the video it was writing. Delete that one before resubmitting -- the writer
# does not go through a temporary file, so a truncated video would be mistaken
# for a finished one.
#
# Scoring is deliberately not here: it needs torchvision, which the generating
# environment does not have to have. See slurm/now_score.sbatch.
#
# The first config decides the baseline, which every row on the same prompt set
# and backbone shares.
# =============================================================================

#SBATCH --job-name=fomo_rows
#SBATCH --output=logs/rows_%j.out
#SBATCH --error=logs/rows_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=122400M

# No -e: one broken row must not take the others down with it.
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export REPO_DIR="$PWD"
# shellcheck source=slurm/env.sh
source "$REPO_DIR/slurm/env.sh"
require_modules torch diffusers peft

: "${CONFIGS:?set CONFIGS=\"experiments/<a>.yaml experiments/<b>.yaml\"}"
read -r -a CONFIG_LIST <<< "$CONFIGS"
for config in "${CONFIG_LIST[@]}"; do
    [ -f "$config" ] || { echo "ERROR: no such experiment: $config" >&2; exit 2; }
done

stage() { echo; echo "=== $* ==="; date; }

columns_of() {
    python3 -c "
from experiments import config as ec
print('\n'.join(r.concept for r in ec.rows(ec.load('$1'))))"
}

method_of() {
    python3 -c "
from experiments import config as ec
print(ec.load('$1')['method'])"
}

# The directory the pinned checkpoint of one concept would be in, or the
# adapter directory itself when the experiment pins nothing.
checkpoint_of() {
    python3 -c "
import os
from experiments import config as ec, layout
cfg = ec.load('$1')
directory = layout.adapter_dir(cfg['name'], '''$2''',
                               cfg['train'].get('mask_mode', 'unclamped'),
                               cfg['train']['imap_alpha'])
pinned = cfg['train'].get('checkpoint')
print(os.path.join(directory, pinned) if pinned else directory)"
}

stage "baseline for ${CONFIG_LIST[0]}"
python3 -m experiments.generate "${CONFIG_LIST[0]}" --base || echo "FAILED baseline"

for config in "${CONFIG_LIST[@]}"; do
    if [ "$(method_of "$config")" = "fomo" ]; then
        IDX=0
        while read -r concept; do
            [ -n "$concept" ] || continue
            checkpoint=$(checkpoint_of "$config" "$concept")
            if [ -f "$checkpoint/pytorch_lora_weights.safetensors" ]; then
                echo "trained already: $concept"
            else
                stage "train $concept  ($config)"
                # A child bash, because train.sbatch ends in exec and would
                # otherwise replace this script.
                CONFIG="$config" SLURM_ARRAY_TASK_ID="$IDX" \
                    bash "$REPO_DIR/slurm/train.sbatch" \
                    || echo "FAILED training $concept"
            fi
            IDX=$((IDX + 1))
        done < <(columns_of "$config")
    fi

    stage "columns of $config"
    python3 -m experiments.generate "$config" --all || echo "FAILED columns of $config"
done

stage "done"
