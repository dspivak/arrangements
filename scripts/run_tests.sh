#!/bin/bash
# Run the test suite one file at a time, printing a line per file as it lands.
#
# The whole suite in one command takes about an hour under load, so any timeout on it is a
# silent cut-off partway through (the book repo's AGENTS.md, "The limit is the threat, never
# the schedule"). One file at a time finishes, and is read, well inside a limit.
#
#   scripts/run_tests.sh                 # every tests/test_*.py
#   scripts/run_tests.sh 300             # with a 300s limit per file (default 540)
#   scripts/run_tests.sh 540 tests/test_cascade.py tests/test_ratchet.py
#
# Prints "file  seconds  verdict" per file and a summary; exits nonzero if any file failed or
# was cut off, and names which.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LIMIT=${1:-540}
shift || true
FILES=("$@")
[ ${#FILES[@]} -eq 0 ] && FILES=(tests/test_*.py)
bad=()
for f in "${FILES[@]}"; do
  start=$(date +%s)
  out=$(timeout "$LIMIT" $PY -m pytest "$f" -q 2>&1 | tail -3 | tr '\n' ' ')
  code=$?
  secs=$(( $(date +%s) - start ))
  if [ $code -eq 124 ]; then
    verdict="CUT OFF at ${LIMIT}s -- split this file with -k, or raise the limit"
    bad+=("$f (cut off)")
  elif [ $code -ne 0 ]; then
    verdict="FAILED :: $out"
    bad+=("$f (failed)")
  else
    verdict="$(echo "$out" | grep -oE '[0-9]+ passed[^|]*' | head -1)"
  fi
  printf '%-44s %5ds  %s\n' "$f" "$secs" "$verdict"
done
if [ ${#bad[@]} -eq 0 ]; then
  echo "all ${#FILES[@]} files passed"
else
  printf 'NOT CLEAN: %s\n' "${bad[*]}"
  exit 1
fi
