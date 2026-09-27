#!/bin/bash
# Run an explicitly selected batch; return to the caller between batches.
# Inspect progress and split a slow file by pytest node ID before its limit is near.
#   scripts/run_tests.sh --list
#   scripts/run_tests.sh 120 tests/test_polynomial.py
#   scripts/run_tests.sh 120 tests/test_file.py::test_name
# Full output is streamed and saved; the exit status records any failure or cutoff.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
if [ "${1:-}" = "--list" ] && [ "$#" -eq 1 ]; then
  printf '%s\n' tests/test_*.py
  exit 0
fi
if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: scripts/run_tests.sh LIMIT_SECONDS TEST_FILE_OR_NODE_ID [...]" >&2
  echo "Use --list to choose a small batch; estimate its duration before running." >&2
  exit 64
fi
LIMIT=$1
shift
if [ ! -x "$PY" ]; then
  echo "Missing $PY; install the test environment first (README.md)." >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "The timeout command is required." >&2
  exit 1
fi
LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/arrangements-tests.XXXXXX") || exit 1
printf 'Full logs: %s\n' "$LOGDIR"
bad=()
index=0
for selection in "$@"; do
  index=$((index + 1))
  logfile="$LOGDIR/$index.log"
  start=$(date +%s)
  printf '\nSTART %s (limit %ss)\n' "$selection" "$LIMIT"
  timeout "$LIMIT" "$PY" -u -m pytest "$selection" -v 2>&1 | tee "$logfile"
  statuses=("${PIPESTATUS[@]}")
  code=${statuses[0]}
  secs=$(( $(date +%s) - start ))
  if [ "${statuses[1]}" -ne 0 ]; then
    verdict="LOG WRITE FAILED"
    bad+=("$selection (log failure)")
  elif [ "$code" -eq 124 ]; then
    verdict="CUT OFF at ${LIMIT}s; split this selection before retrying"
    bad+=("$selection (cut off)")
  elif [ "$code" -ne 0 ]; then
    verdict="FAILED (exit $code)"
    bad+=("$selection (failed)")
  else
    verdict="PASSED"
  fi
  printf '%s  %ss  %s  log=%s\n' "$selection" "$secs" "$verdict" "$logfile"
  # A failure or cutoff needs inspection before starting another selected piece.
  [ ${#bad[@]} -ne 0 ] && break
done
if [ ${#bad[@]} -eq 0 ]; then
  printf 'All %s selected pieces passed.\n' "$index"
else
  printf 'NOT CLEAN: %s\n' "${bad[*]}"
  exit 1
fi
