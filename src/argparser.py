# Copyright (c) 2026 EPFL
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
from pathlib import Path
import subprocess
import math


def parse_simulation_args() -> argparse.Namespace:
    """
    Parse command-line arguments for cohesive-element simulations.
    Constructs an argument parser, processes user inputs, resolves file paths,
    generates mesh files if necessary, and creates a unique run identifier
    based on simulation parameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - Simulation parameters (length, time, strain rate, etc.)
        - File paths (mesh_file, material_file, output_root)
        - Run identifier (id)
        - Derived values (n_elements, mesh_variation, etc.)

    Raises
    ------
    subprocess.CalledProcessError
        If mesh generation script fails.
    FileNotFoundError
        If input directory structure is missing.

    Notes
    -----
    Mesh files are auto-generated if they don't exist, based on the
    combination of length, n_elements, element_order, and variation parameters.
    """
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Resolve paths and generate derived values
    args = _resolve_paths(args)
    args = _generate_mesh_info(args)
    args = _generate_run_id(args)
    return args


def _create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser with all simulation parameters.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all CLI arguments defined.
    """
    parser = argparse.ArgumentParser(
        description="Explicit cohesive-element simulation for 1D bar fracture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ─── Output Configuration ───

    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-root",
        "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Base output directory (defaults to 'output/' in project root)",
    )
    output_group.add_argument(
        "--id",
        "-id",
        type=str,
        default=None,
        metavar="ID",
        help="Custom run identifier (overrides auto-generated ID)",
    )
    output_group.add_argument(
        "--study-name",
        "-sn",
        type=str,
        default=None,
        metavar="NAME",
        help="Optional study name; if given, outputs are placed under output-root/study-name/run-id/",
    )
    output_group.add_argument(
        "--n-dumps",
        "-d",
        type=int,
        default=200,
        metavar="INT",
        help="Number of output dumps during simulation",
    )

    # ─── Geometry & Mesh ───

    mesh_group = parser.add_argument_group("Geometry & Mesh")
    mesh_group.add_argument(
        "--length",
        "-l",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="Specimen length in meters",
    )
    mesh_group.add_argument(
        "--n-elements",
        "-n",
        type=int,
        default=1000,
        metavar="INT",
        help="Number of finite elements",
    )
    mesh_group.add_argument(
        "--mesh-density",
        "-md",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Mesh density multiplier (overrides n-elements if != 1.0)",
    )
    mesh_group.add_argument(
        "--mesh-variation",
        "-v",
        type=float,
        default=0.4,
        metavar="FLOAT",
        help="Random mesh variation fraction in [0, 0.5)",
    )
    mesh_group.add_argument(
        "--mesh-element-order",
        "-p",
        type=int,
        choices=[1, 2],
        default=1,
        metavar="INT",
        help="Mesh polynomial order (1=linear, 2=quadratic)",
    )

    # ─── Time & Loading ───

    time_group = parser.add_argument_group("Time & Loading")
    time_group.add_argument(
        "--total-time",
        "-t",
        type=float,
        default=1e-6,
        metavar="FLOAT",
        help="Total simulation time in seconds",
    )
    time_group.add_argument(
        "--safety-factor",
        "-s",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="CFL stability safety factor (0 < s ≤ 1)",
    )
    time_group.add_argument(
        "--strain-rate-factor",
        "-r",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Strain-rate multiplier for loading",
    )
    time_group.add_argument(
        "--impact-velocity",
        "-iv",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Initial impact velocity (for run_impact.py)",
    )

    # ─── Contact & Cohesive Parameters ───

    contact_group = parser.add_argument_group("Contact & Cohesive Parameters")
    contact_group.add_argument(
        "--contact-type",
        "-c",
        type=str,
        choices=["penalty", "nonsmooth"],
        default="penalty",
        metavar="STR",
        help="Contact formulation type",
    )
    contact_group.add_argument(
        "--contact-factor",
        "-con",
        type=float,
        default=10.0,
        metavar="FLOAT",
        help="Contact penalty factor α: k_con = α × (E/h)",
    )
    contact_group.add_argument(
        "--cohesive-factor",
        "-coh",
        type=float,
        default=float("inf"),
        metavar="FLOAT",
        help="Cohesive stiffness factor: 0=auto, inf=off, >0=specific (nonsmooth mode)",
    )
    contact_group.add_argument(
        "--restitution",
        "-e",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Coefficient of restitution (nonsmooth mode, 0 ≤ e ≤ 1)",
    )

    # ─── Material Defects ───

    defect_group = parser.add_argument_group("Material Defects")
    defect_group.add_argument(
        "--defects-density",
        "-dd",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Density of initial defects (defects per unit length)",
    )
    defect_group.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="INT",
        help="Random seed for strength distribution and mesh variation",
    )

    # ─── Boundary Conditions & Constraints ───

    bc_group = parser.add_argument_group("Boundary Conditions & Constraints")
    bc_group.add_argument(
        "--apply-bc",
        "-b",
        action="store_true",
        help="Apply displacement boundary conditions at edges",
    )
    bc_group.add_argument(
        "--box",
        action="store_true",
        help="Constrain fragment motion within a bounding box",
    )
    bc_group.add_argument(
        "--box-size-factor",
        "-bf",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="Box size factor relative to critical deformation + travel distance",
    )

    # ─── Advanced Options ───

    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--cohesive-insertion-ratio",
        "-cir",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="Fraction of potential cohesive elements to insert (0.0–1.0)",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Resolve input/output paths and select material file based on contact type.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with output_root and other parameters.

    Returns
    -------
    argparse.Namespace
        Updated args with:
        - input_root : Path to input directory
        - output_root : Resolved output directory path
        - material_file : Path to selected material file
    """
    project_root = Path(__file__).resolve().parent.parent

    # Set input directory
    args.input_root = project_root / "input"

    # Resolve output directory
    if args.output_root:
        # Keep the literal "." so that scripts can write directly into the
        # current directory when the launcher is executed from inside a run.
        if str(args.output_root) != ".":
            args.output_root = Path(args.output_root).resolve()
        else:
            args.output_root = Path(args.output_root)
    else:
        args.output_root = project_root / "output"

    # Optional study sub-folder (does not affect the run ID)
    if args.study_name:
        args.output_root = args.output_root / args.study_name

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Select material file based on contact type
    mat_type = "nsn" if args.contact_type == "nonsmooth" else "penalty"
    args.material_file = (
        args.input_root / "material" / f"material_linear_{mat_type}.dat"
    )
    return args


def _generate_mesh_info(args: argparse.Namespace) -> argparse.Namespace:
    """
    Determine mesh parameters and generate mesh file if it doesn't exist.
    Applies safety limits on mesh variation based on element order, computes
    the effective number of elements, and calls the mesh generator if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with mesh-related parameters.

    Returns
    -------
    argparse.Namespace
        Updated args with:
        - mesh_file : Path to the mesh file
        - n_elements : Effective number of elements (int)
        - mesh_variation : Clamped variation value
    """
    mesh_dir = args.input_root / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Clamp mesh variation based on element order (higher order = stricter limits)
    variation_limits = {1: 0.4, 2: 0.2}
    max_variation = variation_limits.get(args.mesh_element_order, 0.4)
    args.mesh_variation = min(args.mesh_variation, max_variation)

    if args.mesh_variation > max_variation:
        print(
            f"Note: Mesh variation capped at {max_variation:.2f} "
            f"for P{args.mesh_element_order} elements"
        )

    # Compute effective number of elements
    if args.mesh_density != 1.0:
        args.n_elements = int(args.length * args.mesh_density)
    else:
        args.n_elements = int(args.n_elements)

    # Build mesh filename
    base_name = (
        f"bar_1D_l{args.length:.2e}_n{args.n_elements:.2e}_"
        f"p{args.mesh_element_order}"
    )

    if args.mesh_variation > 0:
        var_str = f"_var{int(args.mesh_variation * 100)}"
        mesh_name = f"{base_name}{var_str}.msh"
    else:
        mesh_name = f"{base_name}.msh"

    mesh_path = mesh_dir / mesh_name

    # Generate mesh if it doesn't exist
    if not mesh_path.exists():
        print(f"Generating mesh: {mesh_name}")
        _generate_mesh_file(mesh_dir, args)
    args.mesh_file = str(mesh_path)

    return args


def _generate_mesh_file(mesh_dir: Path, args: argparse.Namespace) -> None:
    """
    Call the external mesh generation script.

    Parameters
    ----------
    mesh_dir : Path
        Directory containing generate_mesh.py and output location.
    args : argparse.Namespace
        Arguments containing mesh parameters.

    Raises
    ------
    subprocess.CalledProcessError
        If the mesh generation script fails.
    """

    cmd = [
        "python",
        str(mesh_dir / "generate_mesh.py"),
        "--length",
        str(args.length),
        "--n-elements",
        str(args.n_elements),
        "--element-order",
        str(args.mesh_element_order),
        "--output-dir",
        str(mesh_dir),
    ]

    if args.mesh_variation > 0:
        cmd.extend(["--variation-range", str(args.mesh_variation)])
    subprocess.run(cmd, check=True)


def _generate_run_id(args: argparse.Namespace) -> argparse.Namespace:
    """
    Generate a unique run identifier based on simulation parameters.
    Creates a descriptive ID string that encodes key parameters for easy
    identification of simulation configurations in output files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with all simulation parameters.

    Returns
    -------
    argparse.Namespace
        Updated args with:
        - id : Unique run identifier string
    """
    id_parts = [
        "nsnfrag1d",
        f"l{args.length:.2e}",
        f"n{args.n_elements:.2e}",
        f"md{args.mesh_density:.2e}",
        f"p{args.mesh_element_order}",
        f"t{args.total_time:.2e}",
        f"r{args.strain_rate_factor:.2e}",
        f"s{args.safety_factor:.2e}",
    ]

    # Add defect density if applicable
    if args.defects_density > 0:
        id_parts.append(f"dd{args.defects_density:.2e}")

    # Add contact-specific parameters
    if args.contact_type == "penalty":
        id_parts.append(f"k{args.contact_factor:.2e}")
    else:  # nonsmooth
        id_parts.append(f"e{args.restitution:.3e}")
        if args.cohesive_factor == 0:
            id_parts.append("sc-auto")
        elif not math.isinf(args.cohesive_factor):
            id_parts.append(f"sc-{args.cohesive_factor:.2e}")

    # Add impact velocity if specified
    if args.impact_velocity is not None:
        id_parts.append(f"iv{args.impact_velocity:.2e}")

    # Add boundary condition flags
    if args.apply_bc:
        id_parts.append("bc")
    if args.box:
        id_parts.extend(["box", f"bf{args.box_size_factor:.2e}"])

    # Always include seed for reproducibility
    id_parts.append(f"seed{args.seed}")

    # Use custom ID if provided, otherwise auto-generate
    args.id = args.id if args.id else "_".join(id_parts)

    return args


if __name__ == "__main__":
    # Quick test: print parsed arguments
    test_args = parse_simulation_args()
    print(f"Run ID: {test_args.id}")
    print(f"Mesh: {test_args.mesh_file}")
    print(f"Material: {test_args.material_file}")
    print(f"Output: {test_args.output_root}")
