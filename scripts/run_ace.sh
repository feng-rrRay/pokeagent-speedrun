#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

args=(
  --game "${GAME:-red}"
  --backend "${BACKEND:-gemini}"
  --model-name "${MODEL:-gemini-3.1-pro-preview}"
  --port "${PORT:-2984}"
  --agent-auto
  --scaffold ace
  --run-name "${RUN_NAME:-ace}"
)

backup_state="${BACKUP_STATE-PokemonRed-GBC/red_init.zip}"
[[ -z "${backup_state}" ]] || args+=(--backup-state "${backup_state}")

exec uv run python run.py "${args[@]}" "$@"
