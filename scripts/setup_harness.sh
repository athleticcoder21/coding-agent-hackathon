#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${1:-}"
if [[ -z "$WHEELHOUSE" || ! -d "$WHEELHOUSE" ]]; then
  echo "Usage: scripts/setup_harness.sh /path/to/official-wheelhouse" >&2
  exit 2
fi
WHEELHOUSE="$(cd "$WHEELHOUSE" && pwd)"
cd "$ROOT"
uv sync --locked --inexact
WHEELS=()
for package in swegemma adk_submission adk_eval_core google_adk; do
  WHEEL="$(find "$WHEELHOUSE" -name "${package}-*-py3-none-any.whl" -print -quit)"
  if [[ -z "$WHEEL" ]]; then
    echo "Missing official wheel: ${package} in $WHEELHOUSE" >&2
    exit 2
  fi
  WHEELS+=("$WHEEL")
done
# Install only the official pure-Python harness wheels from the organizer's
# wheelhouse. PyPI supplies their ordinary dependencies on the desktop.
uv pip install --python "$ROOT/.venv/bin/python" "${WHEELS[@]}"
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_project.py"
"$ROOT/.venv/bin/swegemma" --help >/dev/null
"$ROOT/.venv/bin/python" "$ROOT/scripts/validate_official.py" "$ROOT/submission"
echo "Official harness installed in $ROOT/.venv"
