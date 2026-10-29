#!/usr/bin/env bash
# Source from a role wrapper. Python owns validation, logging and continuation.
set -euo pipefail
tuning_launch() {
    local role="$1"; shift
    local python="" root
    local -a args=()
    while (($#)); do
        case "$1" in
            --python) [[ $# -ge 2 ]] || { echo 'Missing --python value' >&2; return 2; }; python="$2"; shift 2 ;;
            *) args+=("$1"); shift ;;
        esac
    done
    [[ -n "$python" ]] || { echo 'Explicit --python /path/to/python is required' >&2; return 2; }
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PYTHONDONTWRITEBYTECODE=1
    "$python" -B "$root/run_scripts/tuning/launch.py" --role "$role" "${args[@]}" || {
        local code=$?; echo 'BATCH FAILED' >&2; return "$code";
    }
}
