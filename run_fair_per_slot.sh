#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-$(command -v python || command -v python3)}
FAIR_PER_SLOT_OUTPUT=${FAIR_PER_SLOT_OUTPUT:-"$ROOT/outputs/fair_per_slot_20261109"}
GPUS=${GPUS:-0,1}
GPU_WORKERS=${GPU_WORKERS:-4}
CPU_WORKERS=${CPU_WORKERS:-20}
TOTAL_HOURS=${TOTAL_HOURS:-31}
PIDFILE="$FAIR_PER_SLOT_OUTPUT/controller.pid"
LOGFILE="$FAIR_PER_SLOT_OUTPUT/controller.log"
CHILD_PID=""

export PYTHON FAIR_PER_SLOT_OUTPUT GPUS GPU_WORKERS CPU_WORKERS TOTAL_HOURS

alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid=$(tr -d '\r\n ' < "$PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'run_fair_per_slot.sh controller'
}

remaining_hours() {
  "$PYTHON" - "$DEADLINE" <<'PY'
import sys, time
left = (float(sys.argv[1]) - time.time()) / 3600.0
print(max(left, 0.01))
PY
}

stop_child() {
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
}

trap stop_child TERM INT

run_stage() {
  local stage=$1
  local hours
  hours=$(remaining_hours)
  echo "===== START STAGE $stage $(date -Is), remaining_hours=$hours ====="
  "$PYTHON" -u -m revision.runner \
    --output "$FAIR_PER_SLOT_OUTPUT" \
    --stage "$stage" \
    --resume \
    --hours "$hours" \
    --gpus "$GPUS" \
    --workers "$GPU_WORKERS" \
    --cpu-workers "$CPU_WORKERS" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  local rc=$?
  CHILD_PID=""
  echo "===== END STAGE $stage rc=$rc $(date -Is) ====="
  return "$rc"
}

run_all() {
  mkdir -p "$FAIR_PER_SLOT_OUTPUT"
  DEADLINE=$("$PYTHON" - "$TOTAL_HOURS" <<'PY'
import sys, time
print(time.time() + float(sys.argv[1]) * 3600.0)
PY
)
  export DEADLINE
  run_stage fair_critics
  run_stage per_slot
  "$PYTHON" -u -m revision.runner --output "$FAIR_PER_SLOT_OUTPUT" --stage report --resume
  echo "===== FAIR CRITICS + PER-SLOT FINISHED $(date -Is) ====="
}

case "${1:-status}" in
  smoke)
    "$PYTHON" -m pytest -q tests/test_revision.py
    "$PYTHON" -u -m revision.runner --quick --output "$ROOT/outputs/fair_per_slot_smoke" \
      --stage fair_critics --gpus "" --cpu-workers 2 --resume
    "$PYTHON" -u -m revision.runner --quick --output "$ROOT/outputs/fair_per_slot_smoke" \
      --stage per_slot --gpus "" --cpu-workers 2 --resume
    echo "SMOKE PASSED"
    ;;
  dry-run)
    mkdir -p "$FAIR_PER_SLOT_OUTPUT"
    "$PYTHON" -u -m revision.runner --output "$FAIR_PER_SLOT_OUTPUT" --stage fair_critics \
      --dry-run --gpus "$GPUS" --workers "$GPU_WORKERS" --cpu-workers "$CPU_WORKERS"
    "$PYTHON" -u -m revision.runner --output "$FAIR_PER_SLOT_OUTPUT" --stage per_slot \
      --dry-run --gpus "$GPUS" --workers "$GPU_WORKERS" --cpu-workers "$CPU_WORKERS"
    ;;
  start)
    mkdir -p "$FAIR_PER_SLOT_OUTPUT"
    if alive; then
      echo "Already running: PID $(cat "$PIDFILE")"
      exit 1
    fi
    nohup bash "$ROOT/run_fair_per_slot.sh" controller >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 3
    if ! alive; then
      tail -n 80 "$LOGFILE"
      exit 1
    fi
    echo "PID $(cat "$PIDFILE")"
    echo "Log: $LOGFILE"
    echo "Output: $FAIR_PER_SLOT_OUTPUT"
    ;;
  controller)
    run_all
    ;;
  status)
    if alive; then echo "Controller: running (PID $(cat "$PIDFILE"))"; else echo "Controller: not running"; fi
    if [[ -f "$FAIR_PER_SLOT_OUTPUT/suite_status.json" ]]; then
      "$PYTHON" - "$FAIR_PER_SLOT_OUTPUT/suite_status.json" <<'PY'
import json, sys
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
      echo "Stop requested; the current training process will stop at its checkpoint boundary."
    else
      echo "Controller is not running."
    fi
    ;;
  report)
    "$PYTHON" -u -m revision.runner --output "$FAIR_PER_SLOT_OUTPUT" --stage report --resume
    echo "Report: $FAIR_PER_SLOT_OUTPUT/report/REVISION_REPORT.md"
    ;;
  *)
    echo "Usage: bash run_fair_per_slot.sh {smoke|dry-run|start|controller|status|stop|report}"
    exit 2
    ;;
esac
