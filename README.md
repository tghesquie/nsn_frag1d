# NS_Frag1D — 1D Cohesive Fracture and Contact (Akantu)

## Overview

* Explicit **1D solid mechanics** simulation of dynamic fragmentation with **extrinsic cohesive elements** and **unilateral contact**.
* Built on **Akantu** (C++/Python) for FE infrastructure; contact schemes:
  * **Penalty-based contact** (tunable stiffness) from Akantu
  * **Nonsmooth contact** (QP solved with **OSQP**) from `solver.py`
* Outputs: **ParaView** files and **HDF5** for post-processing.

---

## Repository Layout  

| Path                       | Description                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `src/run_fragmentation.py` | Simulate 1D fragmentation, free or constrained expansion                    |
| `src/run_impact.py`        | Simulate the impact of a pre-damaged bar on a wall                          |
| `src/solver.py`            | Explicit predictor–corrector, nonsmooth contact (QP via OSQP), utilities    |
| `src/helper.py`            | Time-step estimates, cohesive/contact helpers, I/O dumpers, BC utilities    |
| `src/argparser.py`         | **CLI parser**, input/output roots, material/mesh paths, run-ID builder     |
| `input/material/`          | Akantu material files (`*.dat`)                                             |
| `input/mesh/`              | 1D meshes (`*.msh`), `generate_mesh.py` for automatic mesh generation       |
| `output/`                  | Simulation outputs under `output/<RUN_ID>/`                                 |
| `src/notebooks/`           | Example notebooks for inspecting results/fields                             |
| `benchmarks/`              | Standalone analytical validation notebooks (`bouncing_ball`, `bar_impact`)  |

---

## Environment Setup

The build logic is centralized in **`install.sh`**. It is used both by the Docker build and for local installations, ensuring the same Akantu commit and CMake flags are always applied.

#### Option A: Docker (Recommended)

To ensure strict reproducibility, the project is containerized using Docker. The Dockerfile delegates the entire build process to `install.sh`, which installs system libraries, pins the exact Akantu commit (`22adc1e`), compiles the C++ engine, and creates the Python environment via `uv`.

1. **Build the Docker Image:** From the root of this repository, run:
    ```bash
    docker build -t explicit-nsn:latest .
    ```
    *(Note: This step takes a few minutes as it compiles the Akantu C++ engine from source. Ensure your `.dockerignore` file is present to prevent uploading large local output folders to the build context).

#### Option B: Local Manual Setup

If you prefer a native installation, `install.sh` handles everything automatically on Debian/Ubuntu-based systems.

1. **Prerequisites:** You need Python ≥ 3.11 and a Debian/Ubuntu-based system with `apt-get`.

2. **Run the Install Script:** Execute `./install.sh` from the repository root (ensure it is executable via `chmod +x install.sh`). It automatically installs the system libraries listed in `pkg.txt` (such as `build-essential`, `cmake`, `git`, `gfortran`, `gmsh`, `libboost-dev`, `libeigen3-dev`, `libmumps-seq-dev`, `libblas-dev`, `liblapack-dev`), clones Akantu (commit `22adc1e`), compiles the C++ engine, creates a virtual environment, and installs the package in editable mode.

   > **Note:** If you are not on a Debian-based system, `install.sh` will print the required packages and skip the system installation so you can install their equivalents manually. You can also set the environment variable `SKIP_SYSTEM_DEPS=1` to skip the system package step entirely (e.g., in CI environments where dependencies are pre-installed).

3. **Activate the Environment:** Every time you open a new terminal, load the environment with a single command:

    ```bash
    source env.sh
    ```

    This activates the Python virtual environment (`.venv`) and sources the Akantu build environment automatically.

4. **Verify the Installation:** To check that everything is working, run the quick sanity-check script:

    ```bash
    ./reproduce.sh
    ```

    Ensure it is executable via `chmod +x reproduce.sh`. This executes a lightweight 100-element fragmentation simulation (`test_reproduce`). If it completes without errors and creates files under `output/test_reproduce/`, the installation is functional.

---

## Running Simulations

#### Option A: Via Docker

Because the Docker container is isolated, you must mount your local input and output folders using the `-v` flag so the container can read or save your meshes and save the results to your hard drive.

**Basic Execution**

```bash
docker run --rm \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/input:/app/input \
    explicit-nsn:latest \
    python3 src/run_fragmentation.py --output-root /app/output [OPTIONS]
```
Or `src/run_impact.py` depending on the case of interest. *(The `--rm` flag ensures the container cleans itself up after the simulation finishes).*

**Example 1:** Penalty-based contact, free expansion, large strain rate, small length, high mesh density

```bash
docker run --rm \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/input:/app/input \
    explicit-nsn:latest \
    python3 src/run_fragmentation.py --output-root /app/output --id test_penalty --contact-type penalty --contact-factor 10 --strain-rate-factor 100 --length 5e-4 --mesh-density 1e6 --safety-factor 0.2
```

**Example 2:** Nonsmooth contact, with contact dissipation, medium strain rate, confined expansion, with initial velocity boundary conditions

```bash
docker run --rm \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/input:/app/input \
    explicit-nsn:latest \
    python3 src/run_fragmentation.py --output-root /app/output --id test_nsn --contact-type nonsmooth --restitution 0.9 --cohesive-factor 10 --strain-rate-factor 1 --box-size-factor 10 --box --apply-bc --length 1e-2 --mesh-density 1e5 --safety-factor 0.99 
```

**Example 3:** Impact of a damaged bar on a rigid wall, fully dissipative nonsmooth contact, with a velocity of 10 m/s and 10% of cohesive elements insertion
```bash
docker run --rm \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/input:/app/input \
    explicit-nsn:latest \
    python3 src/run_impact.py --output-root /app/output --id test_impact --contact-type nonsmooth --restitution 0 --cohesive-factor 1  --impact-velocity 10 --cohesive-insertion-ratio 0.1 --total-time 5e-6 --length 1e-2 --mesh-density 1e5 --safety-factor 0.99
```

#### Option B: Locally (Native)

If you have completed the local manual setup, ensure your virtual environment is activated and the Akantu paths are loaded in your terminal first. You can then run the Python scripts directly from the repository root.

**Basic Execution**

```bash
python3 src/run_impact.py [OPTIONS]
```
Or `src/run_fragmentation.py` depending on the case of interest.

**Example 1:** Penalty-based contact, free expansion, large strain rate, small length, high mesh density

```bash
python3 src/run_fragmentation.py --id test_penalty --contact-type penalty --contact-factor 10 --strain-rate-factor 100 --length 5e-4 --mesh-density 1e6 --safety-factor 0.2
```

**Example 2:** Nonsmooth contact, with contact dissipation, medium strain rate, confined expansion, with initial velocity boundary conditions

```bash
python3 src/run_fragmentation.py --id test_nsn --contact-type nonsmooth --restitution 0.9 --cohesive-factor 10 --strain-rate-factor 1 --box-size-factor 10 --box --apply-bc --length 1e-2 --mesh-density 1e5 --safety-factor 0.99 
```

**Example 3:** Impact of a damaged bar on a rigid wall, fully dissipative nonsmooth contact, with a velocity of 10 m/s and 10% of cohesive elements insertion
```bash
python3 src/run_impact.py --id test_impact --contact-type nonsmooth --restitution 0 --cohesive-factor 1 --impact-velocity 10 --cohesive-insertion-ratio 0.1 --total-time 5e-6 --length 1e-2 --mesh-density 1e5 --safety-factor 0.99
```

---


## Inputs

#### Materials
The material file is chosen **automatically** based on the `--contact-type` flag:

- `--contact-type penalty` → `input/material/material_linear_penalty.dat`
- `--contact-type nonsmooth` → `input/material/material_linear_nsn.dat`

The material properties are those of aluminum oxide.

#### Mesh
The **mesh file name and path are derived from the CLI options** and auto-generated if missing.

- **Base directory**: Automatically resolves to `input/mesh/` relative to the project root
- **Controlled by**:
  - `--length`, `-l` (e.g. `0.01`) 
  - `--n-elements`, `-n` (e.g. `1000`) 
  - `--mesh-density`, `-md` (Overrides `n-elements` if set to anything other than `1.0`) 
  - `--mesh-element-order`, `-p` (1 or 2) 
  - `--mesh-variation`, `-v` (Capped at `0.4` for P1 elements, and `0.2` for P2 elements)

The base mesh filename follows this format: 
```
bar_1D_l{length:.2e}_n{n_elements:.2e}_p{mesh_element_order}
```

If variation is applied, `_var{variation*100}` is appended to the name (e.g., `_var40.msh`). If the file does not exist, the CLI automatically calls `generate_mesh.py`:

```bash
python3 input/mesh/generate_mesh.py --length LENGTH --n-elements N_ELEMENTS --element-order MESH_ELEMENT_ORDER --output-dir input/mesh [--variation-range MESH_VARIATION]
```

#### Cohesive strengths
For reproducibility, random variations of cohesive strength and defects location are generated based on the --seed argument.

---

## Outputs

- **Directory**: `output/<RUN_ID>/` (Defaults to `output/` in the project root if `--output-root` is not specified).
- **ParaView**: `output/<RUN_ID>/paraview/`
  - `tension_*` — bulk fields
  - `cohesive_*` — cohesive fields
- **HDF5**: `output/<RUN_ID>/data.h5` with per-step data.

#### Run ID Format

Unless explicitly overridden using the `--id` flag, the Run ID is auto-generated to encode the exact parameters of the simulation.

A generated ID looks like this:
```
nsnfrag1d_l1.00e-02_n1.00e+03_md1.00e+00_p1_t1.00e-06_r1.00e+00_s1.00e-01_k1.00e+01_seed1
```

It dynamically appends flags for defects (`dd`), impact velocity (`iv`), contact/cohesive factors (`k`, `e`, `sc`), and boundary conditions (`bc`, `box`) only if they are actively used.

---

## Key CLI Arguments
All arguments below are defined in `parse_simulation_args()`.

#### Output Configuration

| Flags | Description | Default |
| :--- | :--- | :--- |
| `--output-root`, `-o` | Base output directory | `output/` |
| `--id`, `-id` | Custom run identifier (overrides auto-generated ID) | `None` |
| `--n-dumps`, `-d` | Number of output dumps during simulation | `200` |

#### Geometry & Mesh

| Flags | Description | Default |
| :--- | :--- | :--- |
| `--length`, `-l` | Specimen length in meters | `0.01` |
| `--n-elements`, `-n` | Number of finite elements | `1000` |
| `--mesh-density`, `-md` | Mesh density multiplier (overrides n-elements if != 1.0) | `1.0` |
| `--mesh-variation`, `-v` | Random mesh variation fraction in [0, 0.5) | `0.4` |
| `--mesh-element-order`, `-p` | Mesh polynomial order (1=linear, 2=quadratic) | `1` |

#### Time & Loading

| Flags | Description | Default |
| :--- | :--- | :--- |
| `--total-time`, `-t` | Total simulation time in seconds | `1e-6` |
| `--safety-factor`, `-s` | CFL stability safety factor (0 < s ≤ 1) | `0.2` |
| `--strain-rate-factor`, `-r` | Strain-rate multiplier for loading | `1.0` |
| `--impact-velocity`, `-iv` | Initial impact velocity (for run_impact.py) | `None` |

#### Contact & Cohesive Parameters

| Flags | Description | Default |
| :--- | :--- | :--- |
| `--contact-type`, `-c` | Contact formulation type: penalty or nonsmooth | `penalty` |
| `--contact-factor`, `-con` | Contact penalty factor α: k_con = α × (E/h) | `10.0` |
| `--cohesive-factor`, `-coh` | Cohesive stiffness factor: 0=auto, inf=off, >0=specific | `inf` |
| `--restitution`, `-e` | Coefficient of restitution (nonsmooth mode, 0 ≤ e ≤ 1) | `1.0` |

#### Material Defects & Constraints

| Flags | Description | Default |
| :--- | :--- | :--- |
| `--defects-density`, `-dd` | Density of initial defects per unit length | `0.0` |
| `--seed` | Random seed for strength distribution and mesh variation | `1` |
| `--apply-bc`, `-b` | Apply displacement boundary conditions at edges | `False` |
| `--box` | Constrain fragment motion within a bounding box | `False` |
| `--box-size-factor`, `-bf` | Box size factor relative to critical deformation | `2.0` |
| `--cohesive-insertion-ratio`, `-cir` | Fraction of potential cohesive elements to insert | `0.2` |

---

## Visualization

Simulation outputs are written to the host disk via Docker volume mounts, so you can inspect them with any tool.

#### Option A: ParaView (Field Visualization, No Extra Setup)

Open the ParaView files directly on your host machine:

```bash
paraview output/<RUN_ID>/paraview/
```

- **Bulk fields**: `displacement`, `velocity`, `stress`, `grad_u`

*(If `paraview` is not installed, download it from [paraview.org](https://www.paraview.org/download/)).*

---

#### Option B: Jupyter Lab inside Docker (Recommended for Analysis)

The notebooks require only standard scientific Python packages (`h5py`, `plotly`, `pandas`, etc.) and **do not need the Akantu C++ engine**. Since `jupyterlab` is already installed in the Docker image, you can launch it directly:

```bash
docker run --rm -p 8888:8888 \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/src/notebooks:/app/src/notebooks \
    explicit-nsn:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token=''
```

Then open **http://localhost:8888** in your browser and navigate to `src/notebooks/`.

*Notes:*
- `--NotebookApp.token=''` disables the password token for convenience (remove in production).
- The third `-v` mount is optional; it lets you edit notebooks on the host and see changes immediately inside the container.

---

#### Option C: Local Lightweight Environment (No Docker)

If you prefer to run notebooks outside Docker entirely, create a minimal environment that does **not** need Akantu:

```bash
python3 -m venv .venv_notebook
source .venv_notebook/bin/activate
pip install jupyterlab h5py plotly pandas numpy
jupyter lab src/notebooks/
```

Then open the printed URL in your browser.

**Available Notebooks:**

| Notebook | Purpose |
| :--- | :--- |
| `src/notebooks/visualize.ipynb` | Inspect a single simulation: energy balance, fragment count, stress/space-time diagrams |

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** — see the `LICENSE` file at the repository root for details.