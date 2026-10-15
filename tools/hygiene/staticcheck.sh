#!/bin/sh
# pre-commit's staticcheck ratchet, over the whole device tree as CI runs it.
# Skips with a message when staticcheck is not installed; CI checks it.
command -v staticcheck >/dev/null || { echo "staticcheck not installed, skipped (CI checks it)"; exit 0; }
exec python3 tools/hygiene/ratchet.py staticcheck --cwd device -- staticcheck ./internal/... ./pkg/...
