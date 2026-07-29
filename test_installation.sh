#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "--- Loading Environment ---"
source "$PROJECT_ROOT/env.sh"

echo "--- Running Quick Installation Test Simulation ---"
python "$PROJECT_ROOT/src/run_fragmentation.py" \
    --id test_reproduce \
    --contact-type nonsmooth \
    --restitution 0.9 \
    --cohesive-factor 1 \
    --length 1e-3 \
    --n-elements 100 \
    --safety-factor 0.99 \
    --apply-bc

echo "--- Installation Test Complete ---"
echo "To inspect results:"
echo "  paraview $PROJECT_ROOT/output/test_reproduce/paraview/tension.pvd"


