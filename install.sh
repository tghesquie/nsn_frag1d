#!/bin/bash
# install.sh — Shared installation script for NS_Frag1D
# Works for both local and Docker installations.
# On Debian/Ubuntu-based systems, automatically installs system dependencies.
# On other systems, prints the required packages and continues.

set -e

# --- 1. Git Repositories and Pinned Commits ---
# Your project repository (where the src/ folder lives)
MY_GIT_URL="https://github.com/tghesquie/nsn_frag1d"
MY_PROJECT_COMMIT="74b022f" 

# The external C++ library dependency (Akantu)
AKANTU_GIT_URL="https://gitlab.com/akantu/akantu.git"
AKANTU_COMMIT="${AKANTU_COMMIT:-22adc1e143ca74fdb70af185536d16ff4a3396de}"
AKANTU_DIR="${AKANTU_DIR:-external}"

PROJECT_ROOT=$(pwd)
AKANTU_PATH="$PROJECT_ROOT/$AKANTU_DIR/akantu"

echo "--- Starting Installation ---"

# ----------------------------------------------------------------------------
# 2. Fetch YOUR project source code if missing (DCSM Mode)
# ----------------------------------------------------------------------------
if [ ! -d "$PROJECT_ROOT/src" ]; then
    echo "--- Source code missing. Fetching snapshot from nsn_frag1d Git repository ---"
    
    # Clone to a temporary directory so we don't conflict with existing DCSM files
    git clone "$MY_GIT_URL" repo_tmp
    cd repo_tmp
    git checkout "$MY_PROJECT_COMMIT"
    cd ..

    # Move the actual source code directory to the project root
    mv repo_tmp/src "$PROJECT_ROOT/src"
    
    # Clean up temporary folder
    rm -rf repo_tmp
    echo "--- Source code successfully placed in root ---"
fi

# ----------------------------------------------------------------------------
# 3. Install system dependencies (Debian/Ubuntu)
# ----------------------------------------------------------------------------
echo "--- Checking / Installing System Dependencies ---"

if [ "${SKIP_SYSTEM_DEPS}" = "1" ] || [ "${SKIP_SYSTEM_DEPS}" = "true" ]; then
    echo "SKIP_SYSTEM_DEPS is set — skipping system package installation."
else
    if command -v apt-get &> /dev/null; then
        PKG_FILE="$PROJECT_ROOT/pkg.txt"
        if [ ! -f "$PKG_FILE" ]; then
            echo "Error: Package list file '$PKG_FILE' not found." >&2
            exit 1
        fi

        mapfile -t APT_PACKAGES < <(grep -v '^[[:space:]]*$' "$PKG_FILE")

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
# 4. Clone & Build External Dependency (Akantu)
# ----------------------------------------------------------------------------
mkdir -p "$AKANTU_DIR"
cd "$AKANTU_DIR"

if [ ! -d "akantu" ]; then
    echo "Cloning Akantu dependency..."
    git clone "$AKANTU_GIT_URL"
fi

cd akantu
echo "Checking out pinned Akantu commit: $AKANTU_COMMIT"
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
# 5. Python virtual environment
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

echo "--- Installation Complete ---"
echo "To activate the environment for this terminal session, run:"
echo "  source env.sh"
echo ""
echo "To verify the installation, run: ./reproduce.sh"
