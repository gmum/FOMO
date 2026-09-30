#!/bin/bash
# =============================================================================
# Activate the project's conda environment, or stop the job immediately.
#
#   source benchmark/env.sh          # sourced, never executed
#
# A job must not depend on the submitting shell having the right variables
# exported. Treating an unset CONDA_SH as "fall back to the system python"
# means the job is allocated a GPU, spends a minute starting up, and dies on
# `import torch` — with the real cause several screens above the traceback.
# Here the environment is located by search, and a failure to find it ends the
# job at once with a message naming everything that was tried.
#
# Overrides, both optional:
#   CONDA_SH   path to conda.sh, tried first
#   ENV_NAME   environment name (default: eraser)
#
# Provides:
#   require_modules <name> ...      exit non-zero unless every module imports
# =============================================================================

ENV_NAME="${ENV_NAME:-eraser}"

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
        if [ -n "$base" ]; then
            printf '%s\n' "$base/etc/profile.d/conda.sh"
        fi
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
        # conda.sh and the activation hooks reference unset variables, so -u
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

# Checked with find_spec rather than a real import: importing torch costs about
# twenty seconds and the point here is only to catch the wrong interpreter.
require_modules() {
    if ! python3 - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write(
        "ERROR: missing from the active environment: %s\n"
        "       interpreter: %s\n" % (", ".join(missing), sys.executable))
    sys.exit(1)
PY
    then
        echo "The environment is active but incomplete — install the missing" >&2
        echo "packages, or point ENV_NAME at the environment that has them." >&2
        exit 1
    fi
}
