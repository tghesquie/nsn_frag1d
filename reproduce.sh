#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "--- Loading Environment ---"
source "$PROJECT_ROOT/env.sh"

echo "--- Running Quick Sanity-Check Simulation ---"
python "$PROJECT_ROOT/src/run_fragmentation.py" \
    --id test_quick \
    --contact-type penalty \
    --length 1e-3 \
    --n-elements 100 \
    --total-time 1e-7 \
    --safety-factor 0.2 \
    --n-dumps 10

echo "--- Test Run Complete ---"
echo "To inspect results:"
echo "  paraview $PROJECT_ROOT/output/test_quick/paraview/"
