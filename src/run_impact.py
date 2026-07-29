# Copyright (c) 2026 EPFL
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import logging
import os
import time

import h5py
import numpy as np
from argparser import parse_simulation_args

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    """
    Main driver for explicit CZM simulation of bar impact in 1D.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed simulation arguments.

    Returns
    -------
    None
    """
    # --- Set Akantu random seed ---

    os.environ["AKA_SEED"] = str(args.seed)
    logger.info(f"Set AKA_SEED = {os.environ['AKA_SEED']}")

    # --- Import Akantu and helper functions ---

    # Delayed import to ensure AKA_SEED is set before Akantu initializes its random number generator
    import akantu as aka
    from helper import (
        archive_inputs_and_write_launcher,
        check_active_contacts,
        cohesive_update,
        dump_step_h5,
        init_paraview_dumpers,
        init_uniaxial_velocity_field,
        apply_bc,
        get_static_displacement,
        insert_defects,
        compute_reference_stiffness,
        get_cfl_time_step,
        compute_total_energy,
        apply_wall_contact_forces,
        append_wall_constraints,
    )
    from solver import (
        corrector,
        predictor,
        solve,
        update_residual,
        solve_contact_qp,
        corrector_contact,
        update_contact_dissipation,
    )

    # --- Simulation Setup ---

    logger.info(f"Simulation ID: {args.id}")
    dim = 1

    # Parse material input
    aka.parseInput(str(args.material_file))
    logger.info(f"Using material file: {args.material_file}")

    # Read mesh
    if not os.path.isfile(str(args.mesh_file)):
        logger.error(
            f"Mesh file not found: {args.mesh_file}. "
            "Please ensure Gmsh is installed and input/mesh/generate_mesh.py works."
        )
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")

    mesh = aka.Mesh(spatial_dimension=dim)
    mesh.read(str(args.mesh_file))
    logger.info(f"Using material file: {args.material_file}")

    # Initialize model
    model = aka.SolidMechanicsModelCohesive(mesh)
    model.initFull(_analysis_method=aka._static, _is_extrinsic=True)

    # Solver options
    solver = model.getNonLinearSolver()
    solver.set("max_iterations", 1e3)
    solver.set("threshold", 1e-7)
    opts = aka.ModelSolverOptions()
    model.initNewSolver(aka._explicit_lumped_mass, opts)

    # --- Material Properties and Physics Constants ---

    mat_0 = model.getMaterial(0)
    mat_1 = model.getMaterial(1)

    E = mat_0.getReal("E")  # noqa: N806
    rho = mat_0.getReal("rho")
    sigma_c = (
        mat_1.getReal("sigma_c")
        if mat_1.getReal("sigma_c") != 0
        else np.mean(mat_1.getInternalReal("sigma_c_eff")(aka._segment_2))
    )
    G_c = mat_1.getReal("G_c")  # noqa: N806
    wave_speed = (E / rho) ** 0.5  # longitudinal wave speed

    # Strain rate calculation
    args.strain_rate = args.strain_rate_factor * (
        sigma_c**3 / (E**2 * G_c) * wave_speed
    )
    logger.info(
        f"Physics Constants:\n"
        f"  - Wave Speed (c): {wave_speed:.2e} m/s\n"
        f"  - Strain Rate:    {args.strain_rate:.2e} s⁻¹"
    )

    ub, lb = mesh.getUpperBounds(), mesh.getLowerBounds()
    length = ub[0] - lb[0]
    time_transit = length / wave_speed

    # --- Output directories ---
    if str(args.output_root) == ".":
        output_dir = args.output_root
    else:
        output_dir = os.path.join(args.output_root, args.id)

    pv_dir = os.path.join(output_dir, "paraview")
    os.makedirs(pv_dir, exist_ok=True)
    init_paraview_dumpers(model, pv_dir)

    h5_dir = os.path.join(output_dir, "data")
    os.makedirs(h5_dir, exist_ok=True)
    h5_path = os.path.join(h5_dir, "data.h5")
    with h5py.File(h5_path, "w", libver="latest"):
        pass

    # Archive inputs and write a reproducibility launcher
    archive_inputs_and_write_launcher(args, output_dir, __file__)

    # --- Model Initialization ---

    # Assembly
    model.assembleStiffnessMatrix()
    model.assembleMass()
    dof_manager = model.getDOFManager()
    model.updateAutomaticInsertion()

    # Set the number of defects
    if args.defects_density > 0:
        insert_defects(
            mesh,
            model,
            args.defects_density,
            defect_variation=0.02,  # For [0.98, 1.00] variation in strength
            type="random",
            seed=args.seed,
        )

    # Contact penalties and cohesive stiffness caps
    reference_stiffness = 0.0
    if args.contact_type == "penalty":
        reference_stiffness = compute_reference_stiffness(
            mesh, model, type="mean", factor=args.contact_factor
        )
        model.getMaterial(1).setReal("penalty", reference_stiffness)
        logger.info(f"Contact: Penalty method (k={reference_stiffness:.2e})")

    elif args.contact_type == "nonsmooth":
        if args.cohesive_factor == float("inf"):  # off
            logger.info("Cohesive stiffness capping: OFF")
        elif args.cohesive_factor == 0.0:  # full
            model.getMaterial(1).setReal("stiffness_cap", reference_stiffness)
            logger.info("Cohesive stiffness capping: FULL")
        elif args.cohesive_factor > 0.0:  # specific factor
            reference_stiffness = compute_reference_stiffness(
                mesh, model, type="mean", factor=args.cohesive_factor
            )
            model.getMaterial(1).setReal("stiffness_cap", reference_stiffness)
            logger.info(
                f"Cohesive stiffness capping with user-defined factor {args.cohesive_factor:.2e}: {reference_stiffness:.2e}"
            )

    # --- Static Solve ----

    if args.apply_bc:
        logger.info("Applying boundary conditions")
        blocked_nodes = [0, 1]
        blocked = model.getBlockedDOFs().ravel()
        blocked[[dim * n + d for n in blocked_nodes for d in range(dim)]] = True

        u_edge = get_static_displacement(mesh, model, alpha=0.9)
        model.dump()
        apply_bc(model, u_edge)
        model.solveStep("static")
        model.dump()
        logger.info("Completed static solve")

    # --- Time-Stepping and Convergence Setup ---
    # Compute time‐step limits
    dt_crit_aka = model.getStableTimeStep()
    dt_crit_cfl = get_cfl_time_step(model, added_stiffness=reference_stiffness)
    logger.info(
        f"Critical Δt CFL (aka): {dt_crit_aka:.2e} s | Critical Δt CFL (calc. with added stiffness): {dt_crit_cfl:.2e} s"
    )
    dt_crit = min(dt_crit_aka, dt_crit_cfl)
    dt = dt_crit * args.safety_factor
    model.setTimeStep(dt)
    n_steps = int(args.total_time / dt)
    dump_freq = max(n_steps // args.n_dumps, 1)

    logger.info(
        f"Time Control:\n"
        f"  - Time Step (dt):   {dt:.2e} s\n"
        f"  - Total steps: {n_steps:.2e}"
        f"  - Output Dump Frequency: every {dump_freq} steps"
    )

    # --- Initial Conditions ---
    v_edge = init_uniaxial_velocity_field(mesh, model, args.strain_rate)
    free = ~model.getBlockedDOFs().flatten()
    H, m_inv, free = cohesive_update(model, mesh, dof_manager)  # noqa: N806

    # Initialize tracking variables
    contact_dissipation, external_work, external_work_dt = 0.0, 0.0, 0.0
    nb_inserted, nb_contact = 0, 0
    e_restitution = args.restitution  # restitution coefficient for nonsmooth contact

    # --- Loop for pre-damaging the bar ---
    # We insert cohesive elements progressively until we reach the target insertion ratio or mean damage level.
    dt_insertion_crit = 0.1 * dt
    to_insert = int(args.cohesive_insertion_ratio * (len(mesh.getNodes()) - 2))
    mean_damage = 0.0
    imposed_damage = 1e-3
    while nb_inserted < to_insert or mean_damage <= 0.1 * imposed_damage:

        if args.apply_bc:
            apply_bc(model, v_edge * dt_insertion_crit)

        model.beforeSolveStep()
        predictor(model, dt_insertion_crit)
        r = update_residual(model)
        a_increment = solve(model, m_inv, r)
        corrector(model, a_increment, dt_insertion_crit)
        model.afterSolveStep(True)

        # Cohesive insertion
        if nb_inserted < to_insert:
            inserted = model.checkCohesiveStress()
        else:
            inserted = 0
            delta_max = model.getMaterial(1).getInternalReal("delta max")(
                aka._cohesive_1d_2
            )
            delta_c_eff = model.getMaterial(1).getInternalReal("delta_c_eff")(
                aka._cohesive_1d_2
            )
            delta_max[:] = imposed_damage * delta_c_eff[:]
            mean_damage = np.mean(
                model.getMaterial(1).getInternalReal("damage")(aka._cohesive_1d_2)
            )

        if inserted > 0:
            dt_insertion_crit = 0.01 * dt
            H, m_inv, free = cohesive_update(  # noqa: N806
                model, mesh, dof_manager, build_H=(args.contact_type == "nonsmooth")
            )
            nb_inserted += inserted
            logger.info(f"Inserted {nb_inserted} cohesive elements")

    logger.info(
        f"Inserted {nb_inserted} cohesive elements with mean damage {mean_damage:.2e}"
    )
    model.dump()
    model.dump("cohesive elements")
    model.getBlockedDOFs()[:] = False

    # -- Reset the state u, v, a ---
    model.getDisplacement()[:, 0] = -args.impact_velocity * time_transit
    model.getVelocity()[:, 0] = args.impact_velocity
    model.getAcceleration()[:, 0] = 0.0
    start_time = time.time()

    penalty = reference_stiffness if args.contact_type == "penalty" else None
    wall_gap = 0.0
    initial_energy = compute_total_energy(model, contact_dissipation)
    wall_side = "right"

    # --- Main Simulation Loop ---
    for n in range(n_steps):

        # Predictor / Residual
        model.beforeSolveStep()
        v = model.getVelocity().flatten()
        predictor(model, dt)

        model.getExternalForce()[:] = 0.0
        if args.contact_type == "penalty":
            _ = apply_wall_contact_forces(
                model,
                mesh,
                penalty=penalty,
                wall_gap=wall_gap,
                side=wall_side,
            )

        r = update_residual(model)

        # Solve and corrector
        if args.contact_type == "nonsmooth":
            if nb_inserted > 0:
                active = check_active_contacts(model, H)
            else:
                active = np.array([], dtype=int)

            H_solve, active_solve, _ = append_wall_constraints(  # noqa: N806
                model, mesh, H, active, wall_gap=wall_gap, side=wall_side
            )

            if active_solve.size > 0:
                nb_contact += active_solve.size
                a_increment, p, w = solve_contact_qp(
                    model,
                    active_solve,
                    m_inv,
                    H_solve,
                    r,
                    v,
                    dt,
                    e_restitution,
                    qp_tol_abs=1e-14,
                    qp_tol_rel=1e-14,
                    qp_max_iter=10000,
                )
                corrector_contact(model, a_increment, w, dt)
                model.assembleInternalForces()  # to update cohesive openings
                contact_dissipation += update_contact_dissipation(
                    model, H_solve[active_solve, :], v, p
                )
            else:
                a_increment = solve(model, m_inv, r)
                corrector(model, a_increment, dt)
        else:
            a_increment = solve(model, m_inv, r)
            corrector(model, a_increment, dt)

        # Post-step updates
        model.afterSolveStep(True)

        # Track work for boundary conditions
        external_work_dt = model.getEnergy("external work")
        external_work += external_work_dt

        # Convergence check
        # Exit early if the energy created by contact exceeds 5% of the initial energy
        total_energy = compute_total_energy(model, contact_dissipation)
        if total_energy - initial_energy > 0.05 * initial_energy:
            logger.warning(
                f"Early exit at step {n} due to excessive contact energy: "
                f"{total_energy - initial_energy:.2e} J (>{0.05 * initial_energy:.2e} J)"
            )
            break

        # --- Output ---

        # ParaView dump
        if n % dump_freq == 0:
            model.dump()

        # HDF5 dump
        if n % dump_freq == 0:
            avg_active = nb_contact / (n + 1)
            logger.info(
                f"Time: {n * dt:.2e}s | Inserted: {nb_inserted} | Avg active: {avg_active:.2f}"
            )

            dump_step_h5(
                model,
                mesh,
                n,
                dt,
                contact_dissipation,
                h5_path,
                external_work,
                save_stress=True,
                impact=True,
            )

    # --- Finalize ---
    elapsed = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed:.2f} s")
    with h5py.File(h5_path, "a", libver="latest") as f:
        f.attrs["simulation_time"] = elapsed
        f.attrs["contact_type"] = args.contact_type


if __name__ == "__main__":

    args = parse_simulation_args()
    run(args)
