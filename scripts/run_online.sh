#!/usr/bin/env bash
# Run the frozen four-turn BRIDGE online repair method.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

exec "${PYTHON_BIN}" "${ROOT}/online/run_bridge.py" "$@"
