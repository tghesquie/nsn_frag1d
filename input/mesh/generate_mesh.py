#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import meshio
import numpy as np


def parse_args() -> argparse.Namespace:
    def variation_fraction(value: str) -> float:
        v = float(value)
        if not (0.0 <= v < 0.5):
            raise argparse.ArgumentTypeError(
                "variation_range must be in [0.0, 0.5). " "Use 0 for uniform mesh."
            )
        return v

    def positive_int(value: str) -> int:
        v = int(float(value))  # allow 1e3-style notation
        if v <= 0:
            raise argparse.ArgumentTypeError("number of elements must be > 0")
        return v

    parser = argparse.ArgumentParser(
        description=(
            "Generate a 1D bar mesh with Gmsh.\n"
            "If variation_range > 0, a non-uniform mesh is created "
            "by randomly perturbing node positions."
        )
    )

    parser.add_argument(
        "--length",
        "-l",
        type=float,
        default=1.0e-2,
        help="Bar length in meters (default: 1e-2).",
    )
    parser.add_argument(
        "--n-elements",
        "-n",
        type=positive_int,
        default=1000,
        help="Number of 1D elements (default: 1e3 ≈ 1000).",
    )
    parser.add_argument(
        "--variation-range",
        "-v",
        type=variation_fraction,
        default=0.0,
        help=(
            "Random variation range as a fraction of element size, "
            "in [0, 0.5). 0 = uniform mesh (default: 0)."
        ),
    )
    parser.add_argument(
        "--element-order",
        "-p",
        type=int,
        choices=[1, 2],
        default=1,
        help="Polynomial element order for Gmsh (1 or 2, default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Directory where .geo/.msh files are written (default: current directory).",
    )

    return parser.parse_args()


def build_geo_content(length: float, lc: float, n_el: int, p: int) -> str:
    return f"""// Define the points
Point(1) = {{{-length/2}, 0, 0, {lc}}};
Point(2) = {{{length/2}, 0, 0, {lc}}};

// Define the line
Line(1) = {{1, 2}};

// Mesh transfinite
Transfinite Line{{1}} = {n_el + 1};

// Define physical entities
Physical Point("x0") = {{1}};
Physical Point("xf") = {{2}};
Physical Line("bar") = {{1}};

// Meshing properties
Mesh.Algorithm = 1;
Mesh.ElementOrder = {p};
"""


def generate_uniform_mesh(output_base: Path, length: float, n_el: int, p: int) -> Path:
    lc = length / n_el

    geo_content = build_geo_content(length, lc, n_el, p)
    geo_path = Path(f"{output_base}.geo")
    msh_path = Path(f"{output_base}.msh")
    geo_path.write_text(geo_content)

    cmd = ["gmsh", "-1", str(geo_path), "-o", str(msh_path)]
    proc = subprocess.Popen(cmd)
    proc.wait()

    if proc.returncode:
        raise RuntimeError(
            f"Gmsh failed with return code {proc.returncode}. "
            f"Command: {' '.join(cmd)}"
        )

    print(f"Uniform 1D mesh generated: {msh_path}")
    return msh_path


def generate_nonuniform_mesh(
    msh_path: Path, lc: float, variation_fraction: float
) -> Path:
    np.random.seed(1)  # for reproducibility

    mesh = meshio.read(str(msh_path))

    # Absolute variation range in meters
    variation_range = variation_fraction * lc

    x_coords = mesh.points[:, 0]
    unique_x_coords = np.unique(np.round(x_coords, decimals=12))

    variations = np.random.uniform(
        -variation_range, variation_range, size=unique_x_coords.shape
    )
    x_variation_map = dict(zip(unique_x_coords, variations))

    mesh.points[:, 0] += np.array(
        [x_variation_map[np.round(x, decimals=12)] for x in x_coords]
    )

    percent = int(variation_fraction * 100)
    # The stem already contains 'bar_1D_{L}_{N}_p{P}'
    modified_path = msh_path.with_name(
        f"{msh_path.stem}_var{percent:d}{msh_path.suffix}"
    )
    meshio.write(str(modified_path), mesh, file_format="gmsh22", binary=False)

    print(
        f"Non-uniform 1D mesh generated with ±{variation_fraction:.2f} * h "
        f"variation and saved to {modified_path}"
    )
    return modified_path


def main() -> None:
    args = parse_args()

    length = float(args.length)
    n_el = int(args.n_elements)
    p = int(args.element_order)
    variation_fraction = float(args.variation_range)

    # Variation limit
    if p == 1:
        # P1 safety limit: 0.4
        limit = 0.4
        limit_name = "P1 Safety Limit"
    elif p == 2:
        # P2 safety limit: 0.2
        limit = 0.2
        limit_name = "P2 Safety Limit"

    if variation_fraction > limit:
        print(
            f"Note: Requested variation ({variation_fraction:.2f}) exceeds the "
            f"{limit_name} ({limit:.2f}). "
            f"Using the capped value: {limit:.2f}"
        )
        # Adapt the choice by capping the value
        variation_fraction = limit

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_el_float = float(n_el)
    output_name = f"bar_1D_l{length:.2e}_n{n_el_float:.2e}_p{p}"
    output_base = (args.output_dir / output_name).resolve()

    msh_path = generate_uniform_mesh(output_base, length, n_el, p)

    if variation_fraction > 0.0:
        lc = length / n_el
        generate_nonuniform_mesh(msh_path, lc, variation_fraction)


if __name__ == "__main__":
    main()
