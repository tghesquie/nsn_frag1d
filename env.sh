#!/usr/bin/env bash
# env.sh — source in your current shell:  source env.sh

# Detect script directory (works in bash/zsh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --- Virtual environment ---
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${SCRIPT_DIR}/.venv/bin/activate"
else
  echo ".venv not found. Create it first with:" >&2
  echo "   python -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi

# --- Akantu environment ---
# Allow override via AKANTU_ENV
AKANTU_ENV="${AKANTU_ENV:-${SCRIPT_DIR}/external/akantu/build/akantu_environement.sh}"

if [ ! -f "${AKANTU_ENV}" ]; then
  echo "Akantu environment script not found: ${AKANTU_ENV}" >&2
  echo "Has Akantu been built? If you used a custom path, set it via:" >&2
  echo "   export AKANTU_ENV=/path/to/akantu/build/akantu_environement.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
. "${AKANTU_ENV}" || {
  echo "Failed to source Akantu environment: ${AKANTU_ENV}" >&2
  return 1 2>/dev/null || exit 1
}

echo "Environment loaded (venv + Akantu)."
