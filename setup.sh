#!/bin/bash
# setup.sh — Shared installation script for NS_Frag1D
# Works for both local and Docker installations.
# On Debian/Ubuntu-based systems, automatically installs system dependencies.
# On other systems, prints the required packages and continues.

set -e

AKANTU_COMMIT="${AKANTU_COMMIT:-22adc1e143ca74fdb70af185536d16ff4a3396de}"
AKANTU_DIR="${AKANTU_DIR:-external}"

PROJECT_ROOT=$(pwd)
AKANTU_PATH="$PROJECT_ROOT/$AKANTU_DIR/akantu"

echo "--- Starting Installation ---"

# ----------------------------------------------------------------------------
# 1. Install system dependencies (Debian/Ubuntu)
# ----------------------------------------------------------------------------
echo "--- Checking / Installing System Dependencies ---"

# Allow power users / CI to skip the system step
if [ "${SKIP_SYSTEM_DEPS}" = "1" ] || [ "${SKIP_SYSTEM_DEPS}" = "true" ]; then
    echo "SKIP_SYSTEM_DEPS is set — skipping system package installation."
else
    if command -v apt-get &> /dev/null; then
        APT_PACKAGES=(
            build-essential
            cmake
            git
            gfortran
            gmsh
            libboost-dev
            libeigen3-dev
            libmumps-seq-dev
            libblas-dev
            liblapack-dev
        )

        echo "Updating package index and installing dependencies..."
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update
            apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
            rm -rf /var/lib/apt/lists/*
        elif command -v sudo &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
        else
            echo "Error: apt-get is available, but neither root access nor sudo was found." >&2
            echo "Please install the following packages manually:" >&2
            printf '  - %s\n' "${APT_PACKAGES[@]}" >&2
            exit 1
        fi
    else
        echo "Warning: apt-get not found. You are likely not on a Debian/Ubuntu-based system." >&2
        echo "Please install the equivalent of the following packages manually:" >&2
        echo "  build-essential cmake git gfortran gmsh libboost-dev libeigen3-dev libmumps-seq-dev libblas-dev liblapack-dev" >&2
        echo "Continuing with the remaining installation steps..."
    fi
fi

# ----------------------------------------------------------------------------
# 2. Clone & Build Akantu
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 3. Python virtual environment
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 4. Done
# ----------------------------------------------------------------------------
echo "--- Installation Complete ---"
echo "To run simulations, you must set the following environment variables:"
echo "export PYTHONPATH=$AKANTU_PATH/build/python"
echo "export LD_LIBRARY_PATH=$AKANTU_PATH/build/python:/usr/lib/x86_64-linux-gnu"
echo "source $PROJECT_ROOT/.venv/bin/activate"
