#!/bin/sh
# pre-commit's gofmt hook. Skips with a message when Go is not installed,
# so a contributor working only on Python can still commit; CI checks it.
command -v gofmt >/dev/null || { echo "gofmt not installed, skipped (CI checks it)"; exit 0; }
out=$(gofmt -l "$@")
[ -z "$out" ] || { echo "$out"; echo "run: gofmt -w" "$out"; exit 1; }
