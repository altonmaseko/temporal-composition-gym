#!/usr/bin/env bash
# Source from a role wrapper. Python (launch.py) owns validation, logging and continuation.
# Usage: run_X_stress.sh --python /path/to/python --threads N [launch.py options]
set -euo pipefail
stress_launch() {
    local role="$1"; shift
    local python="" threads="" root
    local -a args=()
    while (($#)); do
        case "$1" in
            --python) [[ $# -ge 2 ]] || { echo 'Missing --python value' >&2; return 2; }; python="$2"; shift 2 ;;
            --threads) [[ $# -ge 2 ]] || { echo 'Missing --threads value' >&2; return 2; }; threads="$2"; args+=("$1" "$2"); shift 2 ;;
            *) args+=("$1"); shift ;;
        esac
    done
    [[ -n "$python" ]] || { echo 'Explicit --python /path/to/python is required' >&2; return 2; }
    [[ -n "$threads" ]] || { echo 'Explicit --threads N is required' >&2; return 2; }
    [[ -x "$python" ]] || { echo "Python not executable: $python" >&2; return 2; }
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    local code=0
    "$python" -B "$root/run_scripts/stress/launch.py" --role "$role" "${args[@]}" || code=$?
    if ((code != 0 && code != 1 && code != 130)); then
        echo 'BATCH FAILED' >&2
    fi
    return "$code"
}
