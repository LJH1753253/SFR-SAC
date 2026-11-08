#!/usr/bin/env bash
# Revision study (IoT-73509 resubmission): CPO, E1-E3, E6, fair critics, per-slot optimizers, model mismatch.
#
#   bash run_revision.sh smoke                 # 10-minute CPU rehearsal of every stage (meaningless numbers)
#   bash run_revision.sh dry-run               # list the plan and the GPU-time estimate
#   bash run_revision.sh start --gpus 0,1 --workers 8 --cpu-workers 8
#   bash run_revision.sh status
#   bash run_revision.sh stop                  # graceful; start again to resume from checkpoints
#   bash run_revision.sh report                # rebuild outputs/revision/report/REVISION_REPORT.md
#   bash run_revision.sh start --stage fair_critics --gpus 0,1   # one stage only
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
PY=${PYTHON:-$(command -v python || command -v python3)}
OUT=${REVISION_OUTPUT:-"$ROOT/outputs/revision"}
ACTION=${1:-status}
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$OUT"
PIDFILE="$OUT/runner.pid"
alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid; pid=$(tr -d '\r\n ' < "$PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'revision.runner'
}
case "$ACTION" in
  smoke)
    "$PY" -m pytest -q tests/test_revision.py
    "$PY" -m revision.runner --quick --output "$ROOT/outputs/revision_quick" --gpus "" --cpu-workers 2 --resume "$@"
    echo "Quick report: $ROOT/outputs/revision_quick/report/REVISION_REPORT.md"
    ;;
  dry-run)
    "$PY" -m revision.runner --output "$OUT" --dry-run "$@"
    ;;
  start)
    if alive; then echo "Already running: PID $(cat "$PIDFILE")"; exit 1; fi
    nohup "$PY" -u -m revision.runner --output "$OUT" --resume "$@" >> "$OUT/nohup.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 3
    if ! alive; then tail -n 40 "$OUT/nohup.log"; exit 1; fi
    echo "PID $(cat "$PIDFILE"); log: $OUT/nohup.log"
    ;;
  status)
    if alive; then echo "Runner: running (PID $(cat "$PIDFILE"))"; else echo "Runner: not running"; fi
    [[ ! -f "$OUT/suite_status.json" ]] || "$PY" - "$OUT/suite_status.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"stage={s.get('stage')} completed={s.get('completed')}/{s.get('planned')} failed={len(s.get('failed', {}))}")
for a in s.get("active", []):
    print(f"  running: {a['task']} {a['variant']} gpu={a['gpu']}")
EOF
    ;;
  stop)
    if alive; then kill -TERM "$(cat "$PIDFILE")"; echo "Stop requested (training checkpoints at episode boundaries)."
    else echo "Runner is not running."; fi
    ;;
  report)
    "$PY" -m revision.runner --output "$OUT" --stage report --resume "$@"
    echo "Report: $OUT/report/REVISION_REPORT.md"
    ;;
  *) echo "Usage: bash run_revision.sh {smoke|dry-run|start|status|stop|report} [options]"; exit 2 ;;
esac
