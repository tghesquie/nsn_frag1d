#!/bin/bash
# setup.sh — Shared installation script for NS_Frag1D
# Works for both local and Docker installations.

set -e

AKANTU_COMMIT="${AKANTU_COMMIT:-22adc1e143ca74fdb70af185536d16ff4a3396de}"
AKANTU_DIR="${AKANTU_DIR:-external}"

PROJECT_ROOT=$(pwd)
AKANTU_PATH="$PROJECT_ROOT/$AKANTU_DIR/akantu"

echo "--- Starting Installation ---"

mkdir -p "$AKANTU_DIR"
cd "$AKANTU_DIR"

if [ ! -d "akantu" ]; then
    echo "Cloning Akantu..."
    git clone https://gitlab.com/akantu/akantu.git
fi

cd akantu
echo "Checking out pinned commit: $AKANTU_COMMIT"
git checkout "$AKANTU_COMMIT"

mkdir -p build && cd build
echo "Configuring Akantu with CMake..."

cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DAKANTU_COHESIVE_ELEMENT=ON \
    -DAKANTU_CONTACT_MECHANICS=ON \
    -DAKANTU_DAMAGE_NON_LOCAL=OFF \
    -DAKANTU_DEBUG_TOOLS=OFF \
    -DAKANTU_DIFFUSION=OFF \
    -DAKANTU_DOCUMENTATION=OFF \
    -DAKANTU_DUMPERS=ON \
    -DAKANTU_EMBEDDED=OFF \
    -DAKANTU_EXAMPLES=OFF \
    -DAKANTU_IMPLICIT=ON \
    -DAKANTU_IMPLICIT_SOLVER="Mumps" \
    -DAKANTU_PARALLEL=OFF \
    -DAKANTU_PHASE_FIELD=OFF \
    -DAKANTU_PYTHON_INTERFACE=ON \
    -DAKANTU_SOLID_MECHANICS=ON \
    -DAKANTU_STRUCTURAL_MECHANICS=OFF \
    -DAKANTU_TESTS=OFF \
    -DAKANTU_TRACTION_AT_SPLIT_NODE_=OFF

echo "Building Akantu (this may take a while)..."
cmake --build . -j 4

cd "$PROJECT_ROOT"
echo "Setting up Python virtual environment..."

if command -v uv &> /dev/null; then
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -e .
else
    echo "uv not found, falling back to standard venv..."
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -e .
fi

echo "--- Installation Complete ---"
echo "To run simulations, you must set the following environment variables:"
echo "export PYTHONPATH=$AKANTU_PATH/build/python"
echo "export LD_LIBRARY_PATH=$AKANTU_PATH/build/python:/usr/lib/x86_64-linux-gnu"
echo "source $PROJECT_ROOT/.venv/bin/activate"
