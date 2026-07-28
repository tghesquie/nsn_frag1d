#!/bin/bash
# install.sh — Shared installation script for NS_Frag1D
# Works for both local and Docker installations.
# On Debian/Ubuntu-based systems, automatically installs system dependencies.
# On other systems, prints the required packages and continues.
#
# The source code and input files are expected to be present alongside this
# script (this is the canonical software repository).

set -e

# --- 1. Git Repositories and Pinned Commits ---
# Fallback upstream repository (used only if src/ or input/ are missing)
MY_GIT_URL="https://github.com/tghesquie/nsn_frag1d"
MY_PROJECT_COMMIT="9b95b55"

# The external C++ library dependency (Akantu)
AKANTU_GIT_URL="https://gitlab.com/akantu/akantu.git"
AKANTU_COMMIT="${AKANTU_COMMIT:-22adc1e143ca74fdb70af185536d16ff4a3396de}"
AKANTU_DIR="${AKANTU_DIR:-external}"

PROJECT_ROOT=$(pwd)
AKANTU_PATH="$PROJECT_ROOT/$AKANTU_DIR/akantu"

echo "--- Starting Installation ---"

# ----------------------------------------------------------------------------
# 2. Fetch source code if missing
# ----------------------------------------------------------------------------
if [ ! -d "$PROJECT_ROOT/src" ] || [ ! -d "$PROJECT_ROOT/input" ]; then
    echo "--- Project files missing. Fetching snapshot from nsn_frag1d Git repository ---"

    # Clone to a temporary directory so we don't conflict with existing files
    git clone "$MY_GIT_URL" repo_tmp
    cd repo_tmp
    git checkout "$MY_PROJECT_COMMIT"
    cd ..

    # Move the source code and input directories to the project root if they aren't there
    [ -d repo_tmp/src ] && [ ! -d "$PROJECT_ROOT/src" ] && mv repo_tmp/src "$PROJECT_ROOT/"
    [ -d repo_tmp/input ] && [ ! -d "$PROJECT_ROOT/input" ] && mv repo_tmp/input "$PROJECT_ROOT/"

    # Safely pull critical configurations to the root if they are not already there
    [ -f repo_tmp/pyproject.toml ] && [ ! -f "$PROJECT_ROOT/pyproject.toml" ] && mv repo_tmp/pyproject.toml "$PROJECT_ROOT/"
    [ -f repo_tmp/uv.lock ] && [ ! -f "$PROJECT_ROOT/uv.lock" ] && mv repo_tmp/uv.lock "$PROJECT_ROOT/"
    [ -f repo_tmp/pkg.txt ] && [ ! -f "$PROJECT_ROOT/pkg.txt" ] && mv repo_tmp/pkg.txt "$PROJECT_ROOT/"

    # Clean up temporary folder
    rm -rf repo_tmp
    echo "--- Project files successfully placed in root ---"
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
if [ "${SKIP_AKANTU}" = "1" ] || [ "${SKIP_AKANTU}" = "true" ]; then
    echo "SKIP_AKANTU is set — skipping Akantu compilation backend."
else
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
fi

# ----------------------------------------------------------------------------
# 5. Python virtual environment
# ----------------------------------------------------------------------------
cd "$PROJECT_ROOT"
echo "Setting up Python virtual environment..."

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 command was not found on this system." >&2
    exit 1
fi

# Explicitly add the local bin path to the current script PATH just in case
export PATH="$HOME/.local/bin:$PATH"

# Handle virtual environment build using system uv, sandboxed uv, or fallback pip
if command -v uv &> /dev/null; then
    echo "Using global 'uv' to create virtual environment..."
    if [ ! -d ".venv" ]; then
        uv venv --python python3.11
    fi
    source .venv/bin/activate
    uv pip install -e .
else
    echo "'uv' not found globally. Attempting to fetch a sandboxed instance..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh &> /dev/null; then
        # Manually force the PATH to recognize the freshly downloaded uv binary
        export PATH="$HOME/.local/bin:$PATH"
        echo "Sandboxed 'uv' loaded successfully."
        if [ ! -d ".venv" ]; then
            uv venv --python python3.11
        fi
        source .venv/bin/activate
        uv pip install -e .
    else
        echo "Could not fetch sandboxed 'uv'. Falling back to standard Python venv tool..."
        if [ ! -d ".venv" ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -e .
    fi
fi

echo "--- Installation Complete ---"
echo "To activate the environment for this terminal session, run:"
echo "  source env.sh"
echo ""
if [ "${SKIP_AKANTU}" != "true" ] && [ "${SKIP_AKANTU}" != "1" ]; then
    echo "To verify the installation, run: ./reproduce.sh"
else
    echo "Akantu was skipped. You can now launch the notebooks using:"
    echo "  jupyter lab notebooks/"
fi
