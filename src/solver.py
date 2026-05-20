from __future__ import annotations
import akantu as aka
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from typing import Tuple


def _free_blocked_dofs(model: aka.SolidMechanicsModelCohesive) -> np.ndarray:
    """
    Separate degrees of freedom into free and blocked sets.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model containing DOF information.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - free : Indices of free degrees of freedom.
        - blocked : Indices of blocked (constrained) degrees of freedom.
    """
    blocked_dofs = model.getBlockedDOFs().ravel().astype(bool)
    free = np.flatnonzero(~blocked_dofs)
    blocked = np.flatnonzero(blocked_dofs)

    return free, blocked


def predictor(model: aka.SolidMechanicsModelCohesive, dt: float) -> np.ndarray:
    """
    Compute predicted displacements and velocities using current state.
    Uses explicit Newmark-beta scheme (central difference) for prediction.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model containing state vectors.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Predicted velocity vector.
    """
    free, _ = _free_blocked_dofs(model)
    u = model.getDisplacement().ravel()
    v = model.getVelocity().ravel()
    a = model.getAcceleration().ravel()
    u[free] += dt * v[free] + 0.5 * dt**2 * a[free]
    v[free] += dt * a[free]

    return v.copy()


def update_residual(model: aka.SolidMechanicsModelCohesive) -> np.ndarray:
    """
    Update the residual vector in the model.
    Computes r = f_ext + f_int and zeroes out blocked DOFs.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model containing force vectors.

    Returns
    -------
    np.ndarray
        The residual force vector r.
    """
    _, blocked = _free_blocked_dofs(model)
    model.assembleInternalForces()
    f_int = model.getInternalForce().ravel()
    f_ext = model.getExternalForce().ravel()

    r = f_ext + f_int
    r[blocked] = 0.0

    return r


def solve(
    model: aka.SolidMechanicsModelCohesive, m_inv: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """
    Solve for acceleration increment using lumped mass matrix.
    Computes a_increment = M^{-1} r - a.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model containing acceleration state.
    m_inv : np.ndarray
        1-D array with $1/m_i$ (inverse lumped masses).
    r : np.ndarray
        1-D array of residual forces.

    Returns
    -------
    np.ndarray
        The acceleration increment vector a_increment.
    """
    free, _ = _free_blocked_dofs(model)
    a = model.getAcceleration().ravel()
    a_increment = np.zeros_like(a)

    a_increment[free] = m_inv[free] * r[free] - a[free]

    return a_increment


def corrector(
    model: aka.SolidMechanicsModelCohesive,
    a_increment: np.ndarray,
    dt: float,
) -> None:
    """
    Update velocity and acceleration using acceleration increment.
    Updates state variables based on the computed increment.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model to update.
    a_increment : np.ndarray
        The acceleration increment vector.
    dt : float
        Time step size.
    """
    free, _ = _free_blocked_dofs(model)
    a = model.getAcceleration().ravel()
    v = model.getVelocity().ravel()
    a[free] += a_increment[free]
    v[free] += (dt / 2) * a_increment[free]


def _scale_rows_csr(matrix_csr: csr_matrix, s: np.ndarray) -> csr_matrix:
    """
    Return CSR matrix with row i scaled by s[i].
    Performs scaling without forming a diagonal matrix explicitly.

    Parameters
    ----------
    matrix_csr : scipy.sparse.csr_matrix
        Input matrix in CSR format.
    s : np.ndarray
        Scaling factors for each row.

    Returns
    -------
    scipy.sparse.csr_matrix
        Scaled matrix in CSR format.
    """
    matrix = matrix_csr.copy()
    matrix.data *= np.repeat(s, np.diff(matrix.indptr))

    return matrix


def _scale_cols_csc(matrix_csc: csc_matrix, s: np.ndarray) -> csc_matrix:
    """
    Return CSC matrix with column j scaled by s[j].
    Performs scaling without forming a diagonal matrix explicitly.

    Parameters
    ----------
    matrix_csc : scipy.sparse.csc_matrix
        Input matrix in CSC format.
    s : np.ndarray
        Scaling factors for each column.

    Returns
    -------
    scipy.sparse.csc_matrix
        Scaled matrix in CSC format.
    """
    matrix = matrix_csc.copy()
    matrix.data *= np.repeat(s, np.diff(matrix.indptr))

    return matrix


def solve_contact_qp(
    model: aka.SolidMechanicsModelCohesive,
    active: np.ndarray,  # boolean mask or array of active indices
    m_inv: np.ndarray,  # (ndof,) 1-D inverse lumped masses
    H: sp.spmatrix,  # (n_active, ndof) sparse constraint matrix
    r: np.ndarray,  # (n_nodes,) flat residual forces
    v: np.ndarray,  # (n_nodes,) flat velocities
    dt: float,
    e: float,  # restitution (scalar)
    qp_tol_abs: float = 1e-14,
    qp_tol_rel: float = 1e-14,
    qp_max_iter: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build W', b and solve QP: min 0.5 p^T W' p + b^T p, s.t. p >= 0.
    Solves the contact problem using OSQP to find contact impulses.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model.
    active : np.ndarray
        Boolean mask or array of active contact indices.
    m_inv : np.ndarray
        1-D inverse lumped masses (n_dof,).
    H : scipy.sparse.spmatrix
        Sparse constraint matrix (n_cohesive, n_dof).
    r : np.ndarray
        Flat residual forces (n_nodes,).
    v : np.ndarray
        Flat velocities (n_nodes,).
    dt : float
        Time step size.
    e : float
        Coefficient of restitution (scalar).
    qp_tol_abs : float, optional
        Absolute tolerance for QP solver (default: 1e-14).
    qp_tol_rel : float, optional
        Relative tolerance for QP solver (default: 1e-14).
    qp_max_iter : int, optional
        Maximum iterations for QP solver (default: 10000).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - a_increment : Flattened acceleration increment.
        - p : Flattened contact impulses.
        - w : Flattened velocity correction.
    """
    free, _ = _free_blocked_dofs(model)
    model.assembleStiffnessMatrix()
    K = csc_matrix(aka.AkantuSparseMatrix(model.getDOFManager().getMatrix("K")))
    K = K[free, :][:, free]  # free DOFs only
    H = H[active, :][:, free]  # free DOFs only
    a = model.getAcceleration().ravel()
    a_increment = np.zeros_like(a)
    w = np.zeros_like(a)
    a = a[free]
    m_inv = m_inv[free]
    r = r[free]
    v = v[free]

    # Build W, J, B without forming Minv
    H_csc = H.tocsc()
    Hc = _scale_cols_csc(H_csc, m_inv)  # H * diag(m_inv)
    W = (Hc @ H.T).tocsc()  # (n_active x n_active)
    MinvHt = Hc.T.tocsr()  # (ndof x n_active)
    J = (dt * 0.5) * (K @ MinvHt)  # (ndof x n_active)
    B = (dt * 0.5) * H  # (n_active x ndof)

    # rhs q = H @ ((dt/2)*a) + (1+e)*(H @ v)
    q = H @ (0.5 * dt * a) + (1 + e) * (H @ v)

    # Schur complement pieces
    MinvJ = _scale_rows_csr(J.tocsr(), m_inv)  # (ndof x n_active)
    W_prime = W - (B @ MinvJ)  # (n_active x n_active)
    Minvr = m_inv * (-r)  # diag(m_inv) @ r
    b = q - (B @ Minvr)  # (n_active,)

    is_diagonal = W_prime.nnz == W_prime.shape[0]

    if not is_diagonal:
        # QP: min 0.5 p^T W' p + b^T p  s.t. p >= 0
        n_active = H.shape[0]
        A = sp.eye(n_active, format="csc")
        lower = np.zeros(n_active)
        upper = np.full(n_active, np.inf)

        solver = osqp.OSQP()
        solver.setup(
            P=W_prime.tocsc(),
            q=b.copy(),
            A=A,
            l=lower,
            u=upper,
            eps_abs=qp_tol_abs,
            eps_rel=qp_tol_rel,
            max_iter=qp_max_iter,
            verbose=False,
            polish=False,
        )
        res = solver.solve()
        if res.info.status == "solved inaccurate":
            tol = max(res.info.prim_res, res.info.dual_res)
            solver.update_settings(eps_abs=tol, eps_rel=tol)
            res = solver.solve()
        if res.info.status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"OSQP failed with status '{res.info.status}'")

        p = res.x  # (n_active,) >= 0

    else:
        p = np.maximum(-b / W_prime.diagonal(), 0.0)

    # Recover corrections (flat)
    rhs_a = (J @ p) + (-r)  # (ndof,)
    a_increment[free] = -(m_inv * rhs_a) - a  # a_increment = a_{k+1} - a_k
    rhs_w = H.T @ p  # (ndof,)
    w[free] = m_inv * rhs_w  # diag(m_inv) @ rhs_w

    return a_increment, p, w


def corrector_contact(
    model: aka.SolidMechanicsModelCohesive,
    a_increment: np.ndarray,  # (ndof,) flat acceleration increment
    w: np.ndarray,  # (ndof,) flat velocity correction
    dt: float,
) -> None:
    """
    Update velocity and displacement using acceleration increment and velocity correction.
    Applies contact corrections to the model state.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model to update.
    a_increment : np.ndarray
        Flat acceleration increment (n_dof,).
    w : np.ndarray
        Flat velocity correction (n_dof,).
    dt : float
        Time step size.
    """
    free, _ = _free_blocked_dofs(model)
    a = model.getAcceleration().ravel()
    v = model.getVelocity().ravel()
    u = model.getDisplacement().ravel()

    a[free] += a_increment[free]
    v[free] += (dt / 2) * a_increment[free] + w[free]
    u[free] += (dt / 2) * w[free]


def update_contact_dissipation(
    model: aka.SolidMechanicsModelCohesive,
    H: sp.spmatrix,
    v_old: np.ndarray,  # (n_nodes, dim) old velocities
    p: np.ndarray,  # (n_active,) contact impulses
) -> float:
    """
    Update contact dissipation given contact impulses and old/new velocities.
    Computes energy dissipation D = -p^T v_avg.

    Parameters
    ----------
    model : aka.SolidMechanicsModelCohesive
        The solid mechanics model containing current velocities.
    H : scipy.sparse.spmatrix
        Constraint matrix.
    v_old : np.ndarray
        Old velocities (n_nodes * dim).
    p : np.ndarray
        Contact impulses (n_active,).

    Returns
    -------
    float
        The calculated contact dissipation energy.
    """
    v = model.getVelocity().ravel()
    v_avg = H @ (0.5 * (v + v_old))

    return -p.dot(v_avg)
