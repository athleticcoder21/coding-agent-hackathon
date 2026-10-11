#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="${1:-}"
SUBMISSION_DIR="${2:-$ROOT/submission}"
if [[ -z "$TASK_ID" ]]; then
  echo "Usage: scripts/evaluate.sh INSTANCE_ID [submission_directory]" >&2
  exit 2
fi
DATA="$ROOT/data/competition"
if [[ ! -f "$DATA/tasks.jsonl" || ! -d "$DATA/snapshots" ]]; then
  echo "Official competition tasks/snapshots missing under $DATA" >&2
  exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/swegemma" ]]; then
  echo "Install official harness with scripts/setup_harness.sh first" >&2
  exit 2
fi
"$ROOT/.venv/bin/swegemma" eval \
  --tasks "$DATA/tasks.jsonl" \
  --snapshots-dir "$DATA/snapshots" \
  --submission-dir "$SUBMISSION_DIR" \
  --results-dir "$ROOT/results/$TASK_ID" \
  --sandbox docker \
  --task-id "$TASK_ID" \
  --max-tool-calls 50 \
  --max-time-minutes 30 \
  --concurrency 1 \
  --display auto
