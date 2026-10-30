#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
PY=${PYTHON:-$(command -v python || command -v python3)}
OUT=${ORIGINAL_OUTPUT:-"$ROOT/outputs/original_ch5"}
ACTION=${1:-status}
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$OUT"
PIDFILE="$OUT/suite.pid"
alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid=$(tr -d '\r\n ' < "$PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'original_ch5.runner'
}
case "$ACTION" in
  smoke)
    "$PY" scripts/train_one.py --method sfr_sac --N 6 --seed 0 \
      --episodes 2 --eval-episodes 2 --horizon 12 --device cpu \
      --output "$ROOT/outputs/smoke_sfr_sac"
    ;;
  dry-run)
    "$PY" -m original_ch5.runner --output "$ROOT/outputs/original_ch5_plan" --resume --dry-run "$@"
    ;;
  start)
    if alive; then echo "Already running: PID $(cat "$PIDFILE")"; exit 1; fi
    nohup "$PY" -u -m original_ch5.runner --output "$OUT" --resume --hours 24 --workers 8 "$@" \
      >> "$OUT/nohup.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PIDFILE"
    sleep 3
    if ! alive; then tail -n 50 "$OUT/nohup.log"; exit 1; fi
    echo "PID: $pid"
    echo "Log: $OUT/nohup.log"
    echo "Output: $OUT"
    ;;
  status)
    if alive; then echo "Controller: running (PID $(cat "$PIDFILE"))"; else echo "Controller: not running"; fi
    [[ ! -f "$OUT/suite_status.json" ]] || cat "$OUT/suite_status.json"
    [[ ! -f "$OUT/partial_report_latest.md" ]] || head -n 32 "$OUT/partial_report_latest.md"
    ;;
  stop)
    if alive; then
      kill -TERM "$(cat "$PIDFILE")"
      echo "Stop requested; allow up to three minutes for episode-boundary checkpoints."
    else echo "No matching controller is running."; fi
    ;;
  report|export)
    if alive; then echo "Stop the controller before rebuilding/exporting separately."; exit 1; fi
    "$PY" -m original_ch5.runner --output "$OUT" --resume --stage "$ACTION" "$@"
    ;;
  *) echo "Usage: bash run_original.sh {smoke|dry-run|start|status|stop|report|export} [options]"; exit 2 ;;
esac
