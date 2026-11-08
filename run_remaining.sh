#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-$(command -v python || command -v python3)}
REVISION_OUTPUT=${REVISION_OUTPUT:-"$ROOT/outputs/revision_remaining_20261107"}
GPUS=${GPUS:-0,1}
GPU_WORKERS=${GPU_WORKERS:-4}
CPU_WORKERS=${CPU_WORKERS:-20}
PIDFILE="$REVISION_OUTPUT/remaining.pid"
LOGFILE="$REVISION_OUTPUT/remaining.log"

export PYTHON REVISION_OUTPUT GPUS GPU_WORKERS CPU_WORKERS

STAGES=(mismatch cpo horizon penalty_form ap latency)

alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid=$(tr -d '\r\n ' < "$PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'run_remaining.sh run'
}

run_stage() {
  local stage=$1
  echo "===== START STAGE $stage $(date -Is) ====="
  "$PYTHON" -u -m revision.runner \
    --output "$REVISION_OUTPUT" \
    --stage "$stage" \
    --resume \
    --gpus "$GPUS" \
    --workers "$GPU_WORKERS" \
    --cpu-workers "$CPU_WORKERS"
  echo "===== END STAGE $stage $(date -Is) ====="
}

run_all() {
  mkdir -p "$REVISION_OUTPUT"
  for stage in "${STAGES[@]}"; do
    run_stage "$stage"
  done
  "$PYTHON" -u -m revision.runner --output "$REVISION_OUTPUT" --stage report --resume
  echo "===== ALL REMAINING STAGES FINISHED $(date -Is) ====="
}

case "${1:-status}" in
  smoke)
    bash run_revision.sh smoke
    ;;
  dry-run)
    mkdir -p "$REVISION_OUTPUT"
    "$PYTHON" -u -m revision.runner --output "$REVISION_OUTPUT" --dry-run \
      --gpus "$GPUS" --workers "$GPU_WORKERS" --cpu-workers "$CPU_WORKERS"
    ;;
  start)
    mkdir -p "$REVISION_OUTPUT"
    if alive; then
      echo "Already running: PID $(cat "$PIDFILE")"
      exit 1
    fi
    nohup bash "$ROOT/run_remaining.sh" run >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 3
    if ! alive; then
      tail -n 80 "$LOGFILE"
      exit 1
    fi
    echo "PID $(cat "$PIDFILE")"
    echo "Log: $LOGFILE"
    echo "Output: $REVISION_OUTPUT"
    ;;
  run)
    run_all
    ;;
  status)
    if alive; then
      echo "Controller: running (PID $(cat "$PIDFILE"))"
    else
      echo "Controller: not running"
    fi
    if [[ -f "$REVISION_OUTPUT/suite_status.json" ]]; then
      "$PYTHON" - "$REVISION_OUTPUT/suite_status.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"stage={state.get('stage')} completed={state.get('completed')}/{state.get('planned')} failed={len(state.get('failed', {}))}")
for item in state.get("active", []):
    print(f"  running: {item.get('task')} {item.get('variant')} gpu={item.get('gpu')}")
PY
    fi
    ;;
  stop)
    if alive; then
      kill -TERM "$(cat "$PIDFILE")"
      echo "Stop requested; the current runner will finish its checkpoint boundary."
    else
      echo "Controller is not running."
    fi
    ;;
  report)
    "$PYTHON" -u -m revision.runner --output "$REVISION_OUTPUT" --stage report --resume
    echo "Report: $REVISION_OUTPUT/report/REVISION_REPORT.md"
    ;;
  *)
    echo "Usage: bash run_remaining.sh {smoke|dry-run|start|run|status|stop|report}"
    exit 2
    ;;
esac
