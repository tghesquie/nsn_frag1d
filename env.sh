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
if [ -z "${AKANTU_ENV:-}" ]; then
  cat >&2 <<'EOF'
AKANTU_ENV is not set.
Set it to your Akantu environment script, for example:
  export AKANTU_ENV=/path/to/akantu/build/akantu_environment.sh
Then re-run:
  source env.sh
EOF
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "${AKANTU_ENV}" ]; then
  echo "File not found: ${AKANTU_ENV}" >&2
  echo "Please check that AKANTU_ENV points to a valid file." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
. "${AKANTU_ENV}" || {
  echo "Failed to source Akantu environment: ${AKANTU_ENV}" >&2
  return 1 2>/dev/null || exit 1
}

echo "Environment loaded (venv + Akantu)."
