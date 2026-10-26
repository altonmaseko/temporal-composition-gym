#!/usr/bin/env bash
# Shared implementation of the three protocol-v2 baseline launchers.
# Sourced by run_{memoryless,implicit,explicit}_memory_baselines.sh; not run directly.
#
# One training process at a time, in the plan's seed-major order. Each run writes its own
# stdout/stderr log; a failed run is reported immediately, recorded in the batch status
# file, and makes the whole batch exit nonzero. No success summary is printed if any run failed.

set -Eeuo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --python PATH [options]

  --python PATH            Python of the provisioned environment (required; never a bare 'python')
  --mode full|smoke        full = production budgets (default); smoke = short labelled check
  --seeds "1 2 3 4 5"      training seeds (default: the plan's baseline seeds)
  --no-controls            memoryless role only: skip the DQN-TS task-state controls
  --device-policy POLICY   plan (default: colour runs on CUDA and require it, others on CPU) | cuda | cpu
  --threads N              torch/OMP/MKL CPU threads per run (default 4)
  --wandb-mode MODE        online (default for full) | offline (default for smoke) | disabled
  --wandb-project NAME     default: new_experiments
  --wandb-entity NAME      optional W&B entity
  --results-root DIR       default: run_scripts/baseline/results
  --stop-on-failure        stop at the first failed run (default: record it and continue)
  --continue-batch         restart after a crash: skip planned runs that are already complete,
                           resume unfinished runs that have a restart checkpoint, start the rest fresh
  --skip-preflight         do not run preflight.py first (not recommended)
  --dry-run                print the commands without running anything

W&B authentication: run 'wandb login' once on the machine before launching. The launchers
and training code never read or store the API key.
EOF
}

launch_role() {
  local role="$1"; shift
  local script_dir repo_root
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd)"
  repo_root="$(cd -- "$script_dir/../.." && pwd)"

  local py="" mode="full" seeds="" controls=1 device_policy="plan" threads=4
  local wandb_mode="" wandb_project="new_experiments" wandb_entity="" results_root=""
  local stop_on_failure=0 preflight=1 dry_run=0 continue_batch=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --python) py="$2"; shift 2 ;;
      --mode) mode="$2"; shift 2 ;;
      --seeds) seeds="$2"; shift 2 ;;
      --no-controls) controls=0; shift ;;
      --device-policy) device_policy="$2"; shift 2 ;;
      --threads) threads="$2"; shift 2 ;;
      --wandb-mode) wandb_mode="$2"; shift 2 ;;
      --wandb-project) wandb_project="$2"; shift 2 ;;
      --wandb-entity) wandb_entity="$2"; shift 2 ;;
      --results-root) results_root="$2"; shift 2 ;;
      --stop-on-failure) stop_on_failure=1; shift ;;
      --continue-batch) continue_batch=1; shift ;;
      --skip-preflight) preflight=0; shift ;;
      --dry-run) dry_run=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) echo "unknown option: $1" >&2; usage >&2; return 2 ;;
    esac
  done
  if [[ -z "$py" ]]; then
    echo "error: --python PATH is required (the interpreter of the provisioned environment)" >&2
    return 2
  fi
  if [[ "$mode" != "full" && "$mode" != "smoke" ]]; then echo "error: --mode must be full or smoke" >&2; return 2; fi
  case "$device_policy" in plan|cuda|cpu) ;; *) echo "error: --device-policy must be plan, cuda or cpu" >&2; return 2 ;; esac
  if [[ -z "$wandb_mode" ]]; then
    if [[ "$mode" == "full" ]]; then wandb_mode="online"; else wandb_mode="offline"; fi
  fi
  if ! "$py" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "error: '$py' is not a working Python >= 3.10" >&2
    return 2
  fi

  cd "$repo_root"
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
  export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
  local results_dir="${results_root:-$repo_root/run_scripts/baseline/results}"
  local host batch_id batch_dir
  host="$(hostname 2>/dev/null || echo host)"
  batch_id="${role}__${host}__$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
  batch_dir="$results_dir/v2/$mode/_batches/$batch_id"
  export BASELINE_MACHINE_ROLE="$role" BASELINE_BATCH_ID="$batch_id"

  echo "=== protocol v2 baseline batch: role=$role mode=$mode"
  echo "    python:   $("$py" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  echo "    repo:     $repo_root ($(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
  echo "    results:  $results_dir"
  echo "    batch:    $batch_dir"
  echo "    devices:  policy=$device_policy  threads=$threads  W&B: $wandb_mode / $wandb_project ${wandb_entity:+/ $wandb_entity}"

  local -a plan_args=(-m agents.common.baseline_plan list --role "$role" --mode "$mode")
  [[ -n "$seeds" ]] && plan_args+=(--seeds $seeds)
  [[ "$controls" -eq 0 ]] && plan_args+=(--no-controls)
  local run_list
  run_list="$("$py" "${plan_args[@]}")"
  local total
  total="$(printf '%s\n' "$run_list" | grep -c . || true)"
  echo "    runs:     $total (one at a time)"

  local -a wb_args=(--wandb-mode "$wandb_mode" --wandb-project "$wandb_project")
  [[ -n "$wandb_entity" ]] && wb_args+=(--wandb-entity "$wandb_entity")
  local -a root_args=()
  [[ -n "$results_root" ]] && root_args+=(--results-root "$results_root")

  if [[ "$dry_run" -eq 0 ]]; then
    mkdir -p "$batch_dir/logs"
    printf 'index\tenv\talgo\tseed\texit_code\tstarted_utc\tfinished_utc\tstdout\tstderr\n' > "$batch_dir/batch_status.tsv"
    if [[ "$preflight" -eq 1 ]]; then
      echo "--- preflight"
      if ! "$py" "$repo_root/run_scripts/baseline/preflight.py" --role "$role" --mode "$mode" \
            --device-policy "$device_policy" "${wb_args[@]}" --out "$batch_dir/preflight.json"; then
        echo "PREFLIGHT FAILED: no training was started. See the messages above." >&2
        return 3
      fi
    fi
  fi

  local -a failed=()
  local idx=0 interrupted=0 child_rc
  trap 'interrupted=1' INT TERM
  while read -r env algo seed script; do
    [[ -z "${env:-}" ]] && continue
    idx=$((idx + 1))
    local -a dev_args
    case "$device_policy" in
      cpu) dev_args=(--device cpu) ;;
      cuda) dev_args=(--device cuda --require-gpu) ;;
      plan) if [[ "$env" == "colour" ]]; then dev_args=(--device cuda --require-gpu); else dev_args=(--device cpu); fi ;;
    esac
    local -a extra=()
    [[ "$algo" == "dqn_ts" ]] && extra+=(--task-state)
    local action="fresh" action_dir=""
    if [[ "$continue_batch" -eq 1 ]]; then
      local action_line
      action_line="$("$py" "$repo_root/run_scripts/baseline/batch_action.py" --env "$env" --algo "$algo" \
                       --seed "$seed" --mode "$mode" --root "$results_dir")"
      action="${action_line%%$'\t'*}"
      if [[ "$action" != "fresh" ]]; then action_dir="${action_line#*$'\t'}"; fi
    fi
    if [[ "$action" == "skip" ]]; then
      echo "[$idx/$total] $env / $algo / seed $seed  skipped: already complete in $action_dir"
      if [[ "$dry_run" -eq 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$env" "$algo" "$seed" "skip" "" "" "$action_dir" "" \
          >> "$batch_dir/batch_status.tsv"
      fi
      continue
    fi
    [[ "$action" == "resume" ]] && extra+=(--resume "$action_dir")
    local stem
    stem="$batch_dir/logs/$(printf '%02d' "$idx")_${env}__${algo}__s${seed}"
    local -a cmd=("$py" "$repo_root/$script" --env "$env" --seed "$seed" --mode "$mode" "${extra[@]}"
                  "${dev_args[@]}" --threads "$threads" "${wb_args[@]}" "${root_args[@]}"
                  --machine-role "$role" --batch-id "$batch_id"
                  --launcher-stdout "$stem.out" --launcher-stderr "$stem.err")
    if [[ "$dry_run" -eq 1 ]]; then
      printf '%q ' "${cmd[@]}"; printf '\n'
      continue
    fi
    local started
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$idx/$total] $env / $algo / seed $seed  (started $started${action_dir:+, resuming $action_dir})"
    if "${cmd[@]}" < /dev/null > "$stem.out" 2> "$stem.err"; then child_rc=0; else child_rc=$?; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$env" "$algo" "$seed" "$child_rc" "$started" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stem.out" "$stem.err" >> "$batch_dir/batch_status.tsv"
    if [[ "$child_rc" -eq 0 ]]; then
      echo "      ok: $(grep -E '^complete' "$stem.out" | tail -1)"
    else
      echo "      FAILED (exit $child_rc): $env / $algo / seed $seed -- see $stem.err" >&2
      tail -n 5 "$stem.err" >&2 || true
      failed+=("$env/$algo/seed$seed (exit $child_rc)")
      if [[ "$interrupted" -eq 1 || "$child_rc" -eq 130 ]]; then
        echo "batch interrupted" >&2
        break
      fi
      [[ "$stop_on_failure" -eq 1 ]] && break
    fi
  done <<< "$run_list"
  trap - INT TERM

  [[ "$dry_run" -eq 1 ]] && return 0
  local done_count
  done_count="$(($(wc -l < "$batch_dir/batch_status.tsv") - 1))"
  echo "=== batch $batch_id: $done_count of $total runs attempted"
  if [[ "${#failed[@]}" -gt 0 || "$interrupted" -eq 1 || "$done_count" -ne "$total" ]]; then
    echo "BATCH FAILED: ${#failed[@]} failed run(s); $((total - done_count)) not attempted." >&2
    for f in "${failed[@]}"; do echo "  - $f" >&2; done
    echo "Status per run: $batch_dir/batch_status.tsv" >&2
    return 1
  fi
  echo "All $total runs exited 0 (each verified its budget and final evaluation)."
  echo "Check the full matrix with: $py run_scripts/baseline/check_manifest.py --mode $mode"
  return 0
}
