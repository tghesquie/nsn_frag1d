#!/usr/bin/env bash
# env.sh — source in your current shell:  source workflows/env.sh

# Detect script directory (works in bash/zsh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Virtual environment ---
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${PROJECT_ROOT}/.venv/bin/activate"
else
  echo ".venv not found. Create it first with:" >&2
  echo "   python -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi

# --- Akantu environment (optional for notebook-only usage) ---
# Allow override via AKANTU_ENV
AKANTU_ENV="${AKANTU_ENV:-${PROJECT_ROOT}/external/akantu/build/akantu_environement.sh}"

if [ "${SKIP_AKANTU}" = "1" ] || [ "${SKIP_AKANTU}" = "true" ]; then
  echo "SKIP_AKANTU is set — skipping Akantu environment."
  echo "Environment loaded (venv only)."
else
  if [ ! -f "${AKANTU_ENV}" ]; then
    echo "Akantu environment script not found: ${AKANTU_ENV}" >&2
    echo "Has Akantu been built? If you used a custom path, set it via:" >&2
    echo "   export AKANTU_ENV=/path/to/akantu/build/akantu_environement.sh" >&2
    echo "Or set SKIP_AKANTU=1 to use only the Python virtual environment." >&2
    return 1 2>/dev/null || exit 1
  fi

  # shellcheck disable=SC1090
  . "${AKANTU_ENV}" || {
    echo "Failed to source Akantu environment: ${AKANTU_ENV}" >&2
    return 1 2>/dev/null || exit 1
  }

  echo "Environment loaded (venv + Akantu)."
fi
