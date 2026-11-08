#!/bin/bash
# =============================================================================
# Score several rows in a single job, for a QoS that takes no job arrays --
# the counterpart of slurm/now_rows.sbatch.
#
#   sbatch --qos=now --export=ALL,ENV_NAME=/path/to/env,CONFIGS="\
#          experiments/cogvideox_mapping1.yaml experiments/cogvideox_null.yaml" \
#          slurm/now_score.sbatch
#
# This is separate from generation because it needs torchvision, which the
# environment that generates does not have to have. Point ENV_NAME at the
# environment the published tables were scored with, so the classifier is the
# same here as it was there: ResNet-50 IMAGENET1K_V2 against another weight set
# differs by several points of Top-1 on these classes, and a table mixing them
# would compare methods through the classifier.
#
# The baseline is keyed by the prompt set and the backbone rather than by the
# experiment, so it is scored once from the first config. Everything is
# overwritten rather than appended, so re-running costs only time.
# =============================================================================

#SBATCH --job-name=fomo_score
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=122400M

# No -e: one broken column must not take the rest of the table down with it.
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export REPO_DIR="$PWD"
# shellcheck source=slurm/env.sh
source "$REPO_DIR/slurm/env.sh"
# Checked up front, because the alternative is the same import error repeated
# once per column.
require_modules torch torchvision imageio numpy PIL

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

stage "baseline for ${CONFIG_LIST[0]}"
python3 -m evaluation.resnet "${CONFIG_LIST[0]}" --base || echo "FAILED baseline"

for config in "${CONFIG_LIST[@]}"; do
    stage "columns of $config"
    while read -r concept; do
        [ -n "$concept" ] || continue
        echo "--- $concept"
        python3 -m evaluation.resnet "$config" --erased "$concept" \
            || echo "FAILED $concept"
    done < <(columns_of "$config")
done

for config in "${CONFIG_LIST[@]}"; do
    echo
    python3 -m evaluation.resnet "$config" --report
done

stage "done"
