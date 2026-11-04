#!/bin/bash
# =============================================================================
# Everything every job needs: the repository root, the caches, and an activated
# conda environment -- or an immediate stop with a readable reason.
#
#   source slurm/env.sh          # sourced, never executed
#
# A job must not depend on the submitting shell having exported the right
# things. Treating an unset CONDA_SH as "fall back to the system python" means
# the job is allocated a GPU, spends a minute starting up, and dies on
# `import torch` with the real cause several screens above the traceback. Here
# the environment is located by search and a failure names everything tried.
#
# Overrides, all optional:
#   REPO_DIR   repository root (default: the parent of this file)
#   CONDA_SH   path to conda.sh, tried first
#   ENV_NAME   conda environment name (default: fomo)
#
# Provides:
#   require_modules <name> ...      exit non-zero unless every module imports
# =============================================================================

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_DIR
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-$REPO_DIR/.cache/insightface}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.cache/tmp}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Without this the allocator fragments on long runs and a job that fits reports
# an out-of-memory error with gigabytes reserved and unused.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$REPO_DIR/logs" "$TMPDIR" "$TORCH_HOME"

# On a cluster with environment modules, start from a clean set and load CUDA.
# Skipped silently where there is no module system. MODULES accepts several
# names -- ArcFace scoring also wants cuDNN -- and MODULES= opts out entirely,
# for a machine whose toolkit is simply on the path.
#
#   MODULES="CUDA/12.8.0 cuDNN/9.2.1.18-CUDA-12.8.0" sbatch ...
MODULES="${MODULES-CUDA/12.8.0}"
if type module > /dev/null 2>&1; then
    module purge > /dev/null 2>&1 || true
    if [ -n "$MODULES" ]; then
        # Unquoted on purpose: several names are loaded in one call.
        # shellcheck disable=SC2086
        module load $MODULES || {
            echo "ERROR: module load $MODULES failed. Set MODULES to the names" >&2
            echo "your site uses, or MODULES= to skip loading any." >&2
            exit 1
        }
        echo "env: modules $MODULES"
    fi
fi

ENV_NAME="${ENV_NAME:-fomo}"

# Ordered from most to least authoritative: an explicit override, an already
# initialised conda, then the usual install locations.
_conda_sh_candidates() {
    if [ -n "${CONDA_SH:-}" ]; then
        printf '%s\n' "$CONDA_SH"
    fi
    if [ -n "${CONDA_EXE:-}" ]; then
        printf '%s\n' "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    fi
    if command -v conda > /dev/null 2>&1; then
        local base
        base="$(conda info --base 2> /dev/null || true)"
        [ -n "$base" ] && printf '%s\n' "$base/etc/profile.d/conda.sh"
    fi
    # Unset or non-existent roots are skipped rather than listed, so the error
    # message stays readable.
    local root flavour
    for root in "${HOME:-}" "${SCRATCH:-}" "${PLG_GROUPS_STORAGE:-}"; do
        [ -n "$root" ] && [ -d "$root" ] || continue
        for flavour in miniforge3 mambaforge miniconda3 anaconda3 conda; do
            printf '%s\n' "$root/$flavour/etc/profile.d/conda.sh"
        done
    done
}

_activate_env() {
    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] && [ -f "$candidate" ] || continue
        # conda.sh and its activation hooks reference unset variables, so -u
        # has to come off around them.
        set +u
        if . "$candidate" > /dev/null 2>&1 && conda activate "$ENV_NAME" 2> /dev/null; then
            set -u
            echo "env: activated '$ENV_NAME' from $candidate"
            return 0
        fi
        set -u
    done < <(_conda_sh_candidates)
    return 1
}

if ! _activate_env; then
    {
        echo "ERROR: could not activate the conda environment '$ENV_NAME'."
        echo "Tried, in order:"
        _conda_sh_candidates | sed 's/^/  /'
        echo
        echo "Set CONDA_SH to the conda.sh of your installation, or ENV_NAME if"
        echo "the environment is called something else, and resubmit:"
        echo "  sbatch --export=ALL,CONDA_SH=/path/to/conda.sh,... <script>"
    } >&2
    exit 1
fi

echo "env: python $(python3 -c 'import sys; print(sys.version.split()[0])') at $(command -v python3)"
echo "env: repo $REPO_DIR"

# Checked with find_spec rather than a real import: importing torch costs about
# twenty seconds and the point here is only to catch the wrong interpreter.
require_modules() {
    if ! python3 - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write("ERROR: missing from the active environment: %s\n"
                     "       interpreter: %s\n"
                     % (", ".join(missing), sys.executable))
    sys.exit(1)
PY
    then
        echo "The environment is active but incomplete -- install the missing" >&2
        echo "packages, or point ENV_NAME at the environment that has them." >&2
        exit 1
    fi
}

# Loads one experiment file into EXP_* and TRAIN_*/GEN_* variables.
load_experiment() {
    local config="$1"
    [ -f "$config" ] || { echo "ERROR: no such experiment: $config" >&2; exit 2; }
    eval "$(python3 -m experiments.config "$config" --shell)"
}
