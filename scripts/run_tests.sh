#!/bin/bash
# Run an explicitly selected batch; return to the caller between batches.
# Inspect progress and split a slow file by pytest node ID before its limit is near.
#   scripts/run_tests.sh --list        (each file with its last clean time, to size a batch)
#   scripts/run_tests.sh 120 tests/test_polynomial.py
#   scripts/run_tests.sh 120 tests/test_file.py::test_name
#   PYTEST_ADDOPTS="-s" scripts/run_tests.sh 120 tests/test_file.py   (pytest options, e.g. -s, go here)
# Full output is streamed and saved; the exit status records any failure or cutoff.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
# One line per whole-file run: file, seconds, UTC date, PASSED or CUTOFF. Local, gitignored.
DURATIONS=tests/durations.tsv
if [ "${1:-}" = "--list" ] && [ "$#" -eq 1 ]; then
  for f in tests/test_*.py; do
    t='?'
    [ -f "$DURATIONS" ] && t=$(awk -F'\t' -v f="$f" '
      $1 == f && $4 == "PASSED" { t = $2 "s" }
      $1 == f && $4 == "CUTOFF" { t = ">" $2 "s" }
      END { print (t == "" ? "?" : t) }' "$DURATIONS")
    printf '%-44s %s\n' "$f" "$t"
  done
  echo "Last clean whole-file run in seconds; >N was cut off at N; ? never timed on this machine."
  exit 0
fi
if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: scripts/run_tests.sh LIMIT_SECONDS TEST_FILE_OR_NODE_ID [...]" >&2
  echo "Use --list to see each file's last clean time; size the batch from it." >&2
  echo "Pytest options (e.g. -s) go in PYTEST_ADDOPTS, never as arguments." >&2
  exit 64
fi
LIMIT=$1
shift
# A pytest option given as a selection (say -s) would run the whole suite; options go in PYTEST_ADDOPTS.
for selection in "$@"; do
  if [[ "$selection" == -* ]]; then
    echo "Not a test selection: $selection. Pass pytest options through the environment:" >&2
    echo "  PYTEST_ADDOPTS=\"$selection\" scripts/run_tests.sh $LIMIT TEST_FILE_OR_NODE_ID" >&2
    exit 64
  fi
done
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
  # Record whole-file times for --list; a node ID times only part of its file.
  if [[ "$selection" =~ ^tests/test_[A-Za-z0-9_]+\.py$ ]]; then
    case "$verdict" in
      PASSED) printf '%s\t%s\t%s\tPASSED\n' "$selection" "$secs" "$(date -u +%F)" ;;
      CUT*) printf '%s\t%s\t%s\tCUTOFF\n' "$selection" "$LIMIT" "$(date -u +%F)" ;;
    esac 2>/dev/null >> "$DURATIONS"
  fi
  # A failure or cutoff needs inspection before starting another selected piece.
  [ ${#bad[@]} -ne 0 ] && break
done
if [ ${#bad[@]} -eq 0 ]; then
  printf 'All %s selected pieces passed.\n' "$index"
else
  printf 'NOT CLEAN: %s\n' "${bad[*]}"
  exit 1
fi
