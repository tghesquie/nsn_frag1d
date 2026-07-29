# Copyright (c) 2026 EPFL
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import akantu as aka
import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

# ──── Boundaries and initial conditions ────


def init_uniaxial_velocity_field(
    mesh: aka.Mesh,
    model: aka.SolidMechanicsModelCohesive,
    strain_rate: float,
) -> float:
    """
    Applies a linear uniaxial velocity field in x-direction based on mesh length:
      v_x(pos) = (vel_max / L) * x,
    where vel_max = strain_rate * L/2.

    Parameters
    ----------
    mesh : aka.Mesh
        Mesh object with methods getNodes() and getUpperBounds()/getLowerBounds().
    model : aka.SolidMechanicsModelCohesive
        Model object with method getVelocity() that returns a view of the nodal velocity array.
    strain_rate : float
        Strain rate for the uniaxial velocity field.

    Returns
    -------
    float
        Maximum velocity.
    """
    # Gather nodal data
    velocity = model.getVelocity()
    node_coords = mesh.getNodes()
    ub, lb = mesh.getUpperBounds(), mesh.getLowerBounds()
    length = ub[0] - lb[0]
    vel_max = strain_rate * length / 2

    # Assign linear x-velocity
    maxb = max(ub[0], abs(lb[0]))
    for v_node, pos in zip(velocity, node_coords):
        v_node[0] = (vel_max / maxb) * pos[0]

    return vel_max


def apply_bc(
    model: aka.SolidMechanicsModelCohesive,
    u_edge: float,
) -> None:
    """
    Applies the boundary condition by modifying the displacement field directly.
    This function assumes that the first two DOFs correspond to the left and right edge nodes, respectively.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        Model object with methods getBlockedDOFs() and getDisplacement() that returns a view of the nodal displacement array.
    u_edge : float
        The displacement to apply at the edges (positive for right edge, negative for left edge).

    Returns
    -------
    None
    """
    # Get the blocked DOFs and displacement array
    blocked = model.getBlockedDOFs().ravel()
    if blocked.sum() == 0:
        return
    u = model.getDisplacement().ravel()

    # Apply BCs: left edge gets -u_edge, right edge gets +u_edge
    u[0] -= u_edge
    u[1] += u_edge


def get_static_displacement(
    mesh: aka.Mesh,
    model: aka.SolidMechanicsModelCohesive,
    alpha: float = 0.8,
) -> float:
    """
    Computes the static displacement required to reach a stress close to the smallest strength of the bar.

    Parameters
    ----------
    mesh : aka.Mesh
        Mesh object with methods getNodes() and getUpperBounds()/getLowerBounds().
    model : aka.SolidMechanicsModelCohesive
        Model object with method getMaterial() that returns material properties.
    alpha : float, optional
        A safety factor that multiplies the strength sigma_c.

    Returns
    -------
    float
        The computed static displacement to apply at the edges.
    """
    # Get material properties and mesh length
    lb, ub = mesh.getLowerBounds(), mesh.getUpperBounds()
    length = ub[0] - lb[0]
    E = model.getMaterial(0).getReal("E")
    sigma_c = np.min(model.getMaterial(1).getInternalReal("sigma_c")(aka._point_1))

    # Compute the static displacement using the formula: u = (alpha * sigma_c * L) / E
    static_displacement = alpha * sigma_c * length / E
    return static_displacement / 2


# ──── Model Setup and Configuration ────


def init_paraview_dumpers(
    model: aka.SolidMechanicsModelCohesive,
    outpath: str,
) -> None:
    """
    Configures the model to dump fields for both bulk and cohesive elements into the given directory under base name 'tension' and 'cohesive'.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The Akantu model to configure.
    outpath : str
        Output directory for ParaView dumps.

    Returns
    -------
    None
    """
    # Set base name and directory
    model.setBaseName("tension")
    model.setDirectory(outpath)

    # Add global field dumpers
    model.addDumpFieldVector("displacement")
    model.addDumpFieldVector("external_force")
    model.addDumpFieldVector("internal_force")
    model.addDumpField("velocity")
    model.addDumpField("grad_u")
    model.addDumpField("stress")

    # Add cohesive-element dumpers
    model.setBaseNameToDumper("cohesive elements", "cohesive")
    model.addDumpFieldVectorToDumper("cohesive elements", "displacement")
    model.addDumpFieldToDumper("cohesive elements", "damage")
    model.addDumpFieldVectorToDumper("cohesive elements", "tractions")
    model.addDumpFieldVectorToDumper("cohesive elements", "opening")


def compute_reference_stiffness(
    mesh: aka.Mesh,
    model: aka.SolidMechanicsModelCohesive,
    type: str = "mean",
    factor: float = 10.0,
) -> float:
    """
    Compute a reference stiffness for 1D elements based on mean element length.

    Parameters
    ----------
    mesh : aka.Mesh
        Mesh object with methods getNodes() and getConnectivities().
    model : aka.SolidMechanicsModelCohesive
        Model object with material properties.
    type : str, optional
        Type of length to use for penalty computation ("mean" or "min").
    factor : float, optional
        A multiplicative safety factor for the stiffness.

    Returns
    -------
    float
        Computed reference stiffness value.
    """

    # Get node coordinates and element connectivities
    nodes_coords = mesh.getNodes()
    connectivities = mesh.getConnectivities()

    lengths = []

    # Loop over all 1D element types in the mesh
    for el_type in connectivities.elementTypes(dim=1):
        conns = connectivities(el_type)
        if conns.size == 0:
            continue  # Skip empty element groups

        # Compute lengths of all elements for this type
        segment_lengths = np.linalg.norm(
            nodes_coords[conns[:, 1]] - nodes_coords[conns[:, 0]], axis=1
        )
        lengths.extend(segment_lengths)

    # Check if any valid lengths were computed
    if not lengths:
        raise ValueError("No valid element lengths found.")

    # Compute mean or min element length
    if type == "mean":
        effective_length = np.mean(lengths)
    elif type == "min":
        effective_length = np.min(lengths)
    else:
        raise ValueError(f"Unknown type '{type}'. Use 'mean' or 'min'.")

    # Retrieve Young's modulus from material properties
    E = model.getMaterial(0).getReal("E")

    # Compute penalty using stiffness formula: k = EA/L assuming unit area A=1
    stiffness = E / effective_length * factor

    return stiffness


def insert_defects(
    mesh: aka.Mesh,
    model: aka.SolidMechanicsModelCohesive,
    defects_density: float,
    defect_variation: float = 0.01,
    type: str | None = "random",
    seed: int = 1,
) -> None:
    """
    Maps physical defects (strength reductions) to nearest mesh nodes on a 1D bar.

    Parameters
    ----------
    mesh : aka.Mesh
        The mesh object containing node coordinates.
    model : aka.SolidMechanicsModelCohesive
        The model containing cohesive material properties.
    defects_density : float
        Number of defects per unit length.
    defect_variation : float, optional
        Relative strength reduction (0.01 = 1% reduction).
    type : str | None, optional
        Distribution type ("random" or uniform reduction).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    nodes_coords = mesh.getNodes()
    x_coords = nodes_coords[:, 0]

    low_b = mesh.getLowerBounds()[0]
    high_b = mesh.getUpperBounds()[0]
    bar_length = high_b - low_b

    # Calculate number of defects
    n_defects = int(defects_density * bar_length)
    print(
        f"Generating {n_defects} physical defects on Length={bar_length:.3f}m with seed={seed}"
    )

    # Generate defect positions and strengths
    rng = np.random.default_rng(seed=seed)
    target_positions = rng.uniform(low=low_b, high=high_b, size=n_defects)

    if type == "random":
        target_factors = rng.uniform(
            low=1.0 - defect_variation, high=1.0, size=n_defects
        )
    else:
        target_factors = np.ones(n_defects) * (1.0 - defect_variation)

    # Map defect positions to nearest mesh nodes using binary search
    idx = np.searchsorted(x_coords, target_positions)

    # Clip indices to prevent out-of-bounds
    idx = np.clip(idx, 1, len(x_coords) - 1)

    # Check if the node to the left or right is physically closer
    left_dist = np.abs(x_coords[idx - 1] - target_positions)
    right_dist = np.abs(x_coords[idx] - target_positions)
    nearest_node_indices = np.where(left_dist < right_dist, idx - 1, idx)

    # Apply strengths to Akantu internal fields
    mat = model.getMaterial(1)
    sigma_c_mean = mat.getReal("sigma_c")
    sigma_c = mat.getInternalReal("sigma_c")(aka._point_1)

    # Reset to homogeneous mean first
    sigma_c[:] = sigma_c_mean

    # Use np.minimum.at to handle cases where multiple defects map to the same node
    node_factors = np.ones(x_coords.shape[0])
    np.minimum.at(node_factors, nearest_node_indices, target_factors)

    # Update field
    sigma_c[:] = (sigma_c_mean * node_factors).reshape(-1, 1)

    print(
        f"Mapped {n_defects} defects to mesh. (Effective unique weak nodes: {np.sum(node_factors < 1.0)})"
    )


def restrict_insertion(
    mesh: aka.Mesh,
    model: aka.SolidMechanicsModelCohesive,
    step: int = 1,
) -> None:
    """
    Restricts cohesive element insertion to every nth facet, excluding boundary facets.

    Parameters
    ----------
    mesh : aka.Mesh
        The mesh object.
    model : aka.SolidMechanicsModelCohesive
        The model containing the cohesive element inserter.
    step : int, optional
        Insertion stride (e.g., 2 means every 2nd facet).

    Returns
    -------
    None
    """
    mesh_facets = mesh.getMeshFacets()
    connect_facets = mesh_facets.getConnectivities()
    facets_coords = mesh_facets.getNodes()
    cohesive_inserter = model.getElementInserter()
    check_facets = cohesive_inserter.getCheckFacets()

    for facet_type in connect_facets.elementTypes(dim=0):
        conn_facettype = connect_facets(facet_type)
        check_facettype = check_facets(facet_type)

        # Calculate the midpoints (positions) of each facet
        positions = np.mean(facets_coords[conn_facettype], axis=1)
        x_positions = positions[:, 0]

        # Get the indices that would sort these facets by position
        sorted_indices = np.argsort(x_positions)
        check_facettype[:] = False

        # Enable every 2nd facet
        if len(sorted_indices) > 2:
            internal_indices = sorted_indices[
                1:-1
            ]  # exclude the first and last facet to avoid boundary issues
            # Pick every nth node from the remaining internal list
            active_indices = internal_indices[::step]
            check_facettype[active_indices] = True


# ──── Time Stepping ────


def get_cfl_time_step(
    model: aka.SolidMechanicsModel,
    added_stiffness: float = 0.0,
) -> float:
    """
    Computes a robust stable time step that remains safe even after element insertion/fracture.
    Combines the Bulk CFL limit (based on L_min / c) with an additional stiffness, i.e., contact penalty or cohesive stiffness (acting on the smallest possible nodal mass for safety).

    Parameters
    ----------
    model : aka.SolidMechanicsModel
        The Akantu model containing mesh and material data.
    added_stiffness : float, optional
        The absolute added stiffness at a node [N/m].

    Returns
    -------
    float
        The estimated safe time step.
    """
    mesh = model.getMesh()
    nodes_coords = mesh.getNodes()
    connectivities = mesh.getConnectivities()

    # Find the Minimum Element Length (Global)
    min_length = np.inf

    # Iterate over all 1D element types
    found_elements = False
    for type_index in connectivities.elementTypes(dim=1):
        conns = connectivities(type_index)
        if conns.size == 0:
            continue

        found_elements = True

        # Length computation
        node_start = nodes_coords[conns[:, 0]]
        node_end = nodes_coords[conns[:, 1]]

        lengths = np.linalg.norm(node_end - node_start, axis=1)
        current_min = np.min(lengths)

        if current_min < min_length:
            min_length = current_min

    if not found_elements:
        raise ValueError("No 1D elements found in the mesh to compute time step.")

    # Retrieve Material Properties
    try:
        mat = model.getMaterial(0)
        E = mat.getReal("E")
        rho = mat.getReal("rho")
    except Exception as e:
        print(
            f"Warning: Could not retrieve E/rho from Material 0. Using defaults. Error: {e}"
        )
        E = 1.0
        rho = 1.0

    #  Compute bulk CFL time step
    # c = sqrt(E / rho)
    wave_speed = np.sqrt(E / rho)
    dt_bulk = min_length / wave_speed

    # Compute additional stiffness time step (if applicable)
    if added_stiffness > 1e-12:
        # Calculate the "Worst Case" Mass
        # After fracture, a node might be connected ONLY to the smallest element.
        # Its lumped mass will be 0.5 * Mass_element
        min_element_mass = rho * 1.0 * min_length  # assuming unit cross-sectional area
        min_nodal_mass = 0.5 * min_element_mass

        dt_penalty = 2.0 * np.sqrt(min_nodal_mass / added_stiffness)

        # Combine using Inverse Square Sum
        # 1/dt^2 = 1/dt_bulk^2 + 1/dt_pen^2
        dt_safe = (1.0 / dt_bulk**2 + 1.0 / dt_penalty**2) ** (-0.5)
    else:
        dt_safe = dt_bulk

    return float(dt_safe)


# ──── Linear Algebra Utilities ────


def lump_mass_matrix(
    M: sp.spmatrix | np.ndarray,
) -> sp.spmatrix | np.ndarray:
    """
    Lump a mass matrix by replacing it with a diagonal matrix whose entries
    are the sum of the corresponding rows of M.

    Parameters
    ----------
    M : np.ndarray or scipy.sparse matrix, shape (n, n)
        The original mass matrix.

    Returns
    -------
    M_lumped : same type as M
        The lumped mass matrix, which is diagonal.
    """

    # Check if the matrix is sparse
    if sp.issparse(M):
        # Compute row sums for sparse matrix
        row_sum = np.array(M.sum(axis=1)).ravel()  # (n,)

        # Build diagonal sparse matrix in the same storage format
        M_lumped = sp.diags(row_sum, format=M.format)
    else:
        # Compute row sums for dense matrix
        row_sum = np.sum(M, axis=1)  # (n,)

        # Build dense diagonal matrix
        M_lumped = np.diag(row_sum)

    return M_lumped


def get_free_block_matrix(
    matrix_name: str,
    dof_manager: aka.DOFManager,
    free: Optional[np.ndarray] = None,
    lump: bool = False,
) -> csc_matrix:
    """
    Helper to retrieve any matrix from the DOF manager, optionally lump it,
    and restrict to free DOFs.

    Parameters
    ----------
    matrix_name : str
        Name of the matrix ("M" or "K", etc.) in the DOF manager.
    dof_manager : object
        The model's DOF manager with getMatrix() method.
    free : ndarray of bool, shape (total_DOFs,)
        Mask of free DOFs.
    lump : bool, optional
        If True, lump the raw mass matrix before extracting free block.

    Returns
    -------
    csc_matrix
        The free‐DOF block of the requested matrix.
    """
    raw = csc_matrix(aka.AkantuSparseMatrix(dof_manager.getMatrix(matrix_name)))
    if lump:
        raw = lump_mass_matrix(raw)
    if free is not None:
        return raw[free, :][:, free]
    else:
        return raw


def build_m_inv(
    model: aka.SolidMechanicsModelCohesive, dof_manager: aka.DOFManager
) -> np.ndarray:
    """
    Assemble and return the inverse of the lumped mass matrix as a 1-D array.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model object to assemble mass.
    dof_manager : aka.DOFManager
        The DOF manager for matrix extraction.

    Returns
    -------
    np.ndarray
        1-D array of inverse nodal masses (1/m_i).
    """
    model.assembleMass()
    M = get_free_block_matrix("M", dof_manager, free=None, lump=True)
    m = M.diagonal().astype(np.float64, copy=False)
    return 1.0 / m


# ──── Cohesive Updates ────


def cohesive_update(
    model: aka.SolidMechanicsModelCohesive,
    mesh: aka.Mesh,
    dof_manager: aka.DOFManager,
    H: csr_matrix | None = None,
    build_H: bool = True,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Update cohesive data after potential insertions.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing blocked DOF information.
    mesh : aka.Mesh
        The mesh object for cohesive connectivity.
    dof_manager : aka.DOFManager
        The DOF manager for mass matrix assembly.
    H : csr_matrix | None, optional
        Existing constraint matrix to update incrementally.
    build_H : bool, optional
        If True, rebuild H matrix; otherwise return existing H.

    Returns
    -------
    tuple[csr_matrix, np.ndarray, np.ndarray]
        - H: Updated constraint matrix (CSR)
        - m_inv: 1-D inverse lumped masses for all DOFs
        - free: Boolean mask of free DOFs (length = total DOFs)
    """
    # Free DOFs mask
    blocked = model.getBlockedDOFs().flatten().astype(bool)  # length = total DOFs
    free = ~blocked

    # Inverse lumped mass (1-D)
    m_inv = build_m_inv(model, dof_manager)

    # Updated H (incremental if H provided)
    n_dofs = blocked.size
    if build_H:
        H = build_contact_selection_matrix(mesh, n_dofs, H=H, dim=1)
        return H, m_inv, free
    else:
        return H, m_inv, free


def build_contact_selection_matrix(
    mesh: aka.Mesh, n_dofs: int, H: csr_matrix | None = None, dim: int = 1
) -> csr_matrix:
    """
    Incrementally build/update the cohesive mapping H.
    Assumes cohesive elements are only inserted (appended in connectivity),
    and new DOFs are appended at the end.

    Parameters
    ----------
    mesh : aka.Mesh
        The mesh containing cohesive connectivities.
    n_dofs : int
        Total number of degrees of freedom.
    H : csr_matrix | None, optional
        Existing constraint matrix to update. If None, builds from scratch.
    dim : int, optional
        Spatial dimension (default 1 for 1D problems).

    Returns
    -------
    csr_matrix
        The updated constraint matrix mapping DOFs to cohesive gaps.
    """
    coh_type = mesh.getConnectivities().elementTypes(kind=aka._ek_cohesive)[0]
    cconn_all = np.array(mesh.getConnectivities()(type=coh_type))
    nb_coh_curr = cconn_all.shape[0]

    if H is None:
        # Build from scratch
        return _build_from_connectivity(cconn_all, n_dofs, dim=dim)

    # Incremental update
    nb_coh_old = H.shape[0]
    H_upd = H

    # Grow column count if n_dofs increased
    if H_upd.shape[1] != n_dofs:
        H_upd = H_upd.tocsr()
        H_upd.resize((H_upd.shape[0], n_dofs))

    # Append only newly inserted cohesive elements
    if nb_coh_curr > nb_coh_old:
        cconn_new = cconn_all[nb_coh_old:, :]
        H_rows = _build_from_connectivity(cconn_new, n_dofs, dim=dim)
        H_upd = _append_rows(H_upd, H_rows, n_dofs)

    return H_upd


def _build_from_connectivity(
    cconn_subset: np.ndarray, n_dofs: int, dim: int = 1
) -> csr_matrix:
    """
    Build only the H rows corresponding to the given subset of cohesive connectivity.
    Each row maps nodal DOFs to the mean opening: (-1/n) for face 1 nodes, (+1/n) for face 2 nodes.

    Parameters
    ----------
    cconn_subset : np.ndarray
        Subset of cohesive element connectivity array.
    n_dofs : int
        Total number of degrees of freedom.
    dim : int, optional
        Spatial dimension.

    Returns
    -------
    csr_matrix
        Partial constraint matrix for the given connectivity subset.
    """
    n_new, nn = cconn_subset.shape
    assert nn % 2 == 0, "need an even number of nodes per cohesive element"
    n_face_nodes = nn // 2

    # Row indices for these new cohesive elements
    rows = np.repeat(np.arange(n_new), n_face_nodes)

    # One-DOF-per-node block (dim=1)
    cols_f1 = (dim * cconn_subset[:, :n_face_nodes]).ravel()
    cols_f2 = (dim * cconn_subset[:, n_face_nodes:]).ravel()

    w = 1.0 / float(n_face_nodes)
    n_entries = rows.size

    data = np.empty(2 * n_entries, dtype=float)
    data[:n_entries] = -w
    data[n_entries:] = +w

    rows_full = np.concatenate([rows, rows])
    cols_full = np.concatenate([cols_f1, cols_f2])

    H = coo_matrix((data, (rows_full, cols_full)), shape=(n_new, n_dofs)).tocsr()
    return H


def _append_rows(H_old: csr_matrix, H_new: csr_matrix, n_dofs: int) -> csr_matrix:
    """
    Append new rows to an existing CSR matrix H_old.
    Both matrices must have the same number of columns (n_dofs).

    Parameters
    ----------
    H_old : csr_matrix
        Original constraint matrix.
    H_new : csr_matrix
        New rows to append.
    n_dofs : int
        Number of columns (DOFs) for consistency check.

    Returns
    -------
    csr_matrix
        Vertically stacked matrix [H_old; H_new].
    """
    H = H_old.tocsr()
    if H.shape[1] != n_dofs:
        H.resize((H.shape[0], n_dofs))
    if H_new.shape[1] != n_dofs:
        H_new.resize((H_new.shape[0], n_dofs))

    # Stack vertically and return
    return sp.vstack([H, H_new], format="csr")


# ──── Contact Detection & Wall Constraints ────


def append_wall_constraints(
    model: aka.SolidMechanicsModelCohesive,
    mesh: aka.Mesh,
    H_coh: csr_matrix,
    active_indices: np.ndarray,
    wall_gap: float = 0.0,
    side: str = "right",
) -> Tuple[csr_matrix, np.ndarray, bool]:
    """
    Checks for wall contact and appends wall constraint rows to H if active.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing displacement and velocity data.
    mesh : aka.Mesh
        The mesh containing nodal coordinates.
    H_coh : csr_matrix
        Existing cohesive constraint matrix.
    active_indices : np.ndarray
        Indices of currently active cohesive constraints.
    wall_gap : float, optional
        Gap between wall and nominal boundary position.
    side : str, optional
        Side to check ("left", "right", or "both").

    Returns
    -------
    Tuple[csr_matrix, np.ndarray, bool]
        - Updated constraint matrix (H_coh + wall constraints if active)
        - Updated active indices array
        - Boolean indicating if wall contact was detected
    """

    # Penetration check
    nodes = mesh.getNodes()
    u = model.getDisplacement()[:, 0]
    v = model.getVelocity()[:, 0]
    X = nodes[:, 0]

    new_rows = []
    new_cols = []
    new_data = []
    current_row = 0

    if side in ["right", "both"]:
        idx = int(np.argmax(X))
        wall_pos = X[idx] + wall_gap

        if (X[idx] + u[idx]) >= wall_pos and v[idx] > 0:
            new_rows.append(current_row)
            new_cols.append(idx)
            new_data.append(-1.0)
            current_row += 1

    if side in ["left", "both"]:
        idx = int(np.argmin(X))
        wall_pos = X[idx] - wall_gap

        if (X[idx] + u[idx]) <= wall_pos and v[idx] < 0:
            new_rows.append(current_row)
            new_cols.append(idx)
            new_data.append(+1.0)
            current_row += 1

    if current_row > 0:
        n_dofs = model.getBlockedDOFs().flatten().size

        # Create a sparse matrix for new wall constraints
        H_wall = coo_matrix(
            (new_data, (new_rows, new_cols)), shape=(current_row, n_dofs)
        ).tocsr()

        # Stack with existing H
        H_combined = _append_rows(H_coh, H_wall, n_dofs)

        # Calculate global indices
        n_coh = H_coh.shape[0]
        wall_indices = np.arange(n_coh, n_coh + current_row)

        active_combined = np.concatenate([active_indices, wall_indices]).astype(int)

        return H_combined, active_combined, True

    else:

        return H_coh, active_indices, False


def check_active_contacts(
    model: aka.SolidMechanicsModelCohesive,
    H: csr_matrix,
    v_corr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes contact openings and opening rates to identify active (closing) contacts.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing displacement and velocity data.
    H : csr_matrix
        Constraint matrix mapping DOFs to contact gaps.
    v_corr : Optional[np.ndarray], optional
        Corrected velocity array. If None, uses model velocity.

    Returns
    -------
    np.ndarray
        Flat array of indices where contacts are active (closing).
    """
    u_pred = model.getDisplacement().copy()
    if v_corr is None:
        v_pred = model.getVelocity().copy()
    else:
        v_pred = v_corr.copy()

    # Compute gap openings (negative means penetration/closing)
    opening = H @ u_pred

    # Compute gap opening rates
    opening_rate = H @ v_pred

    # Determine active (closing) contacts
    active_contacts = (opening <= 0) & (opening_rate < 0)

    return np.flatnonzero(active_contacts)


def apply_wall_contact_forces(
    model: aka.SolidMechanicsModelCohesive,
    mesh: aka.Mesh,
    penalty: float,
    wall_gap: float = 1e-7,
    side: str = "right",
) -> bool:
    """
    Applies penalty forces to boundary nodes penetrating rigid walls.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model to apply forces to (modifies external_force).
    mesh : aka.Mesh
        The mesh containing nodal coordinates.
    penalty : float
        Penalty stiffness coefficient [N/m].
    wall_gap : float, optional
        Offset distance from nominal boundary to wall location.
    side : str, optional
        Which wall to check ("left", "right", or "both").

    Returns
    -------
    bool
        True if any contact was detected and forces applied.
    """

    # Get direct view of external forces
    f_ext = model.getExternalForce()
    nodes = mesh.getNodes()[:, 0]
    contact_detected = False

    # Right Wall Logic
    if side in ["right", "both"]:
        idx = int(np.argmax(nodes))

        node_right = mesh.getNodes()[idx, 0]
        u = model.getDisplacement()[idx, 0]

        # Check Penetration
        right_wall_x = mesh.getUpperBounds()[0] + wall_gap
        penetration = (node_right + u) - right_wall_x

        if penetration > 0:
            force = -penalty * penetration
            f_ext[idx, 0] += force
            contact_detected = True

    # Left Wall Logic
    if side in ["left", "both"]:
        idx = int(np.argmin(nodes))

        node_left = mesh.getNodes()[idx, 0]
        u = model.getDisplacement()[idx, 0]

        # Check Penetration
        left_wall_x = mesh.getLowerBounds()[0] - wall_gap
        penetration = left_wall_x - (node_left + u)

        if penetration > 0:
            force = penalty * penetration
            f_ext[idx, 0] += force
            contact_detected = True

    return contact_detected


# ──── Output, Energy & Diagnostics ────


def compute_total_energy(
    model: aka.SolidMechanicsModelCohesive,
    contact_dissipation: float = 0.0,
) -> float:
    """
    Compute the total mechanical energy of the system including dissipation.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing energy data.
    contact_dissipation : float, optional
        Additional dissipated energy from contact.

    Returns
    -------
    float
        Total energy (mechanical + fracture + contact).
    """
    epot = model.getEnergy("potential")
    ekin = model.getEnergy("kinetic")
    edis = model.getEnergy("dissipated")
    erev = model.getEnergy("reversible")
    econ = model.getEnergy("cohesive contact")
    emech = epot + ekin + erev + econ
    total_energy = emech + edis + contact_dissipation
    return total_energy


def _reconstruct_command(args: argparse.Namespace) -> list[str]:
    """
    Rebuild a deterministic CLI command from a parsed argument namespace.

    The returned list excludes the original ``--output-root`` and
    ``--study-name`` because the generated launcher is meant to be executed
    from inside the run directory with ``--output-root .``.
    """
    parts: list[str] = ["--output-root ."]
    if args.id:
        parts.append(f"--id {args.id}")

    parts.extend(
        [
            f"--length {args.length}",
            f"--n-elements {int(args.n_elements)}",
            f"--mesh-density {args.mesh_density}",
            f"--mesh-element-order {args.mesh_element_order}",
            f"--mesh-variation {args.mesh_variation}",
            f"--total-time {args.total_time}",
            f"--safety-factor {args.safety_factor}",
            f"--strain-rate-factor {args.strain_rate_factor}",
            f"--n-dumps {args.n_dumps}",
            f"--contact-type {args.contact_type}",
            f"--defects-density {args.defects_density}",
            f"--seed {args.seed}",
        ]
    )

    if args.contact_type == "penalty":
        parts.extend(
            [
                f"--contact-factor {args.contact_factor}",
                "--restitution 1.0",
                "--cohesive-factor inf",
            ]
        )
    else:  # nonsmooth
        parts.append(f"--restitution {args.restitution}")
        if args.cohesive_factor == 0:
            parts.append("--cohesive-factor 0.0")
        elif not math.isinf(args.cohesive_factor):
            parts.append(f"--cohesive-factor {args.cohesive_factor}")
        else:
            parts.append("--cohesive-factor inf")

    parts.append(f"--cohesive-insertion-ratio {args.cohesive_insertion_ratio}")

    if getattr(args, "impact_velocity", None) is not None:
        parts.append(f"--impact-velocity {args.impact_velocity}")

    if args.apply_bc:
        parts.append("--apply-bc")
    if args.box:
        parts.extend(["--box", f"--box-size-factor {args.box_size_factor}"])

    return parts


def archive_inputs_and_write_launcher(
    args: argparse.Namespace,
    output_dir: str | Path,
    script_path: str | Path,
) -> None:
    """
    Archive the input material and mesh files and write a ``launch.sh``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed simulation arguments.
    output_dir : str | Path
        Run output directory (e.g. ``output/<RUN_ID>/``).
    script_path : str | Path
        Absolute or relative path to the driver script
        (``run_fragmentation.py`` or ``run_impact.py``).

    Notes
    -----
    Creates ``output_dir/input/material/`` and ``output_dir/input/mesh/``
    with copies of the material and mesh files used by the run, plus a
    ``launch.sh`` that reproduces the exact command from the run directory.
    """
    output_dir = Path(output_dir)
    input_root = output_dir / "input"

    material_src = Path(args.material_file)
    mesh_src = Path(args.mesh_file)

    material_dst = input_root / "material" / material_src.name
    mesh_dst = input_root / "mesh" / mesh_src.name
    material_dst.parent.mkdir(parents=True, exist_ok=True)
    mesh_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(material_src, material_dst)
    shutil.copy2(mesh_src, mesh_dst)

    script_path = Path(script_path).resolve()
    cmd_line = " \\\n    ".join(_reconstruct_command(args))

    launcher = (
        "#!/bin/bash\n"
        "\n"
        "# Auto-generated reproducibility launcher\n"
        "# Execute from the directory containing this script.\n"
        "\n"
        'cd "$(dirname "${BASH_SOURCE[0]}")"\n'
        "\n"
        f'python3 "{script_path}" \\\n'
        f"    {cmd_line}\n"
    )

    launch_path = output_dir / "launch.sh"
    launch_path.write_text(launcher)
    launch_path.chmod(0o755)


def _compute_energies(
    model: aka.SolidMechanicsModelCohesive,
    step: int,
    dt: float,
    contact_dissipation: float,
    external_work: float,
) -> dict:
    """
    Extracts and computes all scalar energy metrics including algorithmic energy.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing energy and acceleration data.
    step : int
        Current time step number.
    dt : float
        Time step size.
    contact_dissipation : float
        Contact dissipation energy.
    external_work : float
        Work done by external forces.

    Returns
    -------
    dict
        Dictionary containing time, potential, kinetic, dissipated, reversible,
        contact, algorithmic energies and energy balances.
    """
    a = model.getAcceleration().flatten()
    M = get_free_block_matrix("M", model.getDOFManager(), free=None, lump=True)

    model.assembleMassLumped()
    epot = model.getEnergy("potential")
    ekin = model.getEnergy("kinetic")
    edis = model.getEnergy("dissipated")
    erev = model.getEnergy("reversible")
    econ = model.getEnergy("cohesive contact")

    emech = epot + ekin + erev + econ
    ealgo = emech - (dt**2 / 8) * a.T @ M @ a

    return {
        "time": step * dt,
        "potential_energy": epot,
        "kinetic_energy": ekin,
        "dissipated_energy": edis,
        "reversible_energy": erev,
        "contact_energy": econ,
        "contact_dissipation": contact_dissipation,
        "mechanical_energy": emech,
        "algorithmic_energy": ealgo,
        "mechanical_energy_balance": emech + contact_dissipation + edis - external_work,
        "algorithmic_energy_balance": ealgo
        + contact_dissipation
        + edis
        - external_work,
        "external_work": external_work,
    }


def _init_h5_mesh_data(
    h5f: h5py.File, model: aka.SolidMechanicsModelCohesive, mesh: aka.Mesh
) -> None:
    """
    Handles the one-time dump of global mesh coordinates to HDF5.

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle.
    model : aka.SolidMechanicsModelCohesive
        The model containing the FE engine.
    mesh : aka.Mesh
        The mesh object.

    Returns
    -------
    None
    """
    if "quad_coordinates" in h5f:
        return

    dim = 1
    nb_elements = mesh.getNbElement(type=aka._segment_2)
    nb_quad_per_element = 1

    quad_coords_array = np.zeros(
        nb_elements * nb_quad_per_element * dim, dtype=np.float64
    ).reshape(-1, dim)
    nodes = mesh.getNodes()
    if nodes.ndim == 1:
        nodes = nodes.reshape(-1, dim)

    model.getFEEngine().interpolateOnIntegrationPoints(
        nodes, quad_coords_array, dim, aka._segment_2
    )

    h5f.create_dataset("quad_coordinates", data=quad_coords_array)
    h5f.create_dataset("nodes_coordinates", data=nodes)


def _get_impact_data(model: aka.SolidMechanicsModelCohesive) -> dict:
    """
    Extracts impact-related scalar metrics at the rightmost node.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The model containing displacement, velocity and force data.

    Returns
    -------
    dict
        Dictionary with impact_position, impact_velocity, and impact_force.
    """
    node_right_idx = np.argmin(model.getDisplacement()[:, 0])
    return {
        "impact_position": model.getDisplacement()[node_right_idx, 0],
        "impact_velocity": model.getVelocity()[node_right_idx, 0],
        "impact_force": model.getInternalForce()[node_right_idx, 0],
    }


def dump_step_h5(
    model: aka.SolidMechanicsModelCohesive,
    mesh: aka.Mesh,
    step: int,
    dt: float,
    contact_dissipation: float,
    h5_path: str,
    external_work: float = 0.0,
    save_stress: bool = True,
    impact: bool = False,
) -> None:
    """
    Computes physics step data and appends a group `/step_{step}` to the HDF5 file.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The simulation model.
    mesh : aka.Mesh
        The mesh object.
    step : int
        Current time step index.
    dt : float
        Time step size.
    contact_dissipation : float
        Contact dissipation energy.
    h5_path : str
        Path to the HDF5 output file.
    external_work : float, optional
        Accumulated external work.
    save_stress : bool, optional
        If True, save stress data to file.
    impact : bool, optional
        If True, include impact data in output.

    Returns
    -------
    None
    """

    # Gather data
    attrs = _compute_energies(model, step, dt, contact_dissipation, external_work)

    if impact:
        attrs.update(_get_impact_data(model))

    # Extract arrays
    if save_stress:
        stress_array = (
            model.getMaterial(0).getInternalReal("stress")(aka._segment_2)[:, 0].ravel()
        )

    frag = aka.FragmentManager(model)
    frag.computeAllData()
    nb_fragments = frag.getNbFragment()
    frag_mass = frag.getMass()[:, 0].ravel()
    frag_vel = frag.getVelocity()[:, 0].ravel()

    ###File I/O with retry logic for concurrent access ###
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with h5py.File(h5_path, "a", libver="latest") as h5f:
                h5f.swmr_mode = True

                # One time dump
                if save_stress:
                    _init_h5_mesh_data(h5f, model, mesh)

                # Manage group
                group_name = f"step_{step}"
                if group_name in h5f:
                    print(
                        f"Warning: {group_name} already exists. Deleting to overwrite."
                    )
                    del h5f[group_name]
                grp = h5f.create_group(group_name)

                # Write Scalars (Attributes)
                for key, val in attrs.items():
                    grp.attrs[key] = val
                grp.attrs["nb_fragments"] = nb_fragments

                # Write Heavy Arrays (Datasets)
                if save_stress:
                    grp.create_dataset("stress", data=stress_array, dtype=np.float64)
                grp.create_dataset("fragment_mass", data=frag_mass, dtype=np.float64)
                grp.create_dataset("fragment_velocity", data=frag_vel, dtype=np.float64)

                h5f.flush()
                return  # Success, exit the function

        except BlockingIOError:
            print(f"File locked, retrying in {0.2 * (attempt + 1):.1f}s...")
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))  # exponential backoff
            else:
                print("Max retries reached. Could not write to HDF5 file.")
                return
