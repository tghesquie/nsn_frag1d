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
    Main driver for explicit CZM simulation of dynamic fragmentation in 1D.

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
        check_active_contacts,
        cohesive_update,
        compute_reference_stiffness,
        dump_step_h5,
        init_paraview_dumpers,
        init_uniaxial_velocity_field,
        apply_bc,
        get_static_displacement,
        insert_defects,
        get_cfl_time_step,
        compute_total_energy,
        apply_wall_contact_forces,
        append_wall_constraints,
        restrict_insertion,
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
    logger.info(f"Using mesh file: {args.mesh_file}")

    # Initialize Model
    model = aka.SolidMechanicsModelCohesive(mesh)
    model.initFull(_analysis_method=aka._static, _is_extrinsic=True)

    # Solver Options
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

    # Physics-base convergence criteria
    ub, lb = mesh.getUpperBounds(), mesh.getLowerBounds()
    length = ub[0] - lb[0]
    time_transit = length / wave_speed

    s_qs = 4.0 * E * G_c / (sigma_c**2)
    s_dyn = (24.0 * G_c / (rho * args.strain_rate**2)) ** (1.0 / 3.0)
    s_gc = 1.0 / (1.0 / s_qs + 1.0 / s_dyn)
    s_char = min(s_gc, length)

    time_frag_transit = s_char / wave_speed
    frames_per_transit = 1
    dump_freq = max(1, int(time_frag_transit / (frames_per_transit * dt)))
    max_time = 1e5 * time_transit

    # Convergence check intervals
    conv_check_steps = int(10 * time_frag_transit / dt)
    conv_tol = 1e-3  # relative change in dissipated energy for convergence check

    logger.info(
        f"Time Control:\n"
        f"  - Time Step (dt):   {dt:.2e} s\n"
        f"  - Transit Time (T): {time_transit:.2e} s\n"
        f"  - Max Duration:     {max_time:.2e} s (2.0 * T)\n"
        f"  - Stability Check:  Every {conv_check_steps} steps (1.0 * T)"
        f"  - Output Dump Frequency: every {dump_freq} steps"
    )

    # --- Initial Conditions ---
    v_edge = init_uniaxial_velocity_field(mesh, model, args.strain_rate)
    free = ~model.getBlockedDOFs().flatten()
    H, m_inv, free = cohesive_update(model, mesh, dof_manager)  # noqa: N806
    restrict_insertion(mesh, model, step=1)

    start_time = time.time()
    step, t_current = 0, 0.0

    if args.box:
        wall_gap = (
            (sigma_c * length / (2 * E)) + time_frag_transit * v_edge
        ) * args.box_size_factor
        logger.info(f"Rigid walls enabled with initial gap: {wall_gap:.2e} m")

    # Initialize tracking variables
    contact_dissipation, external_work, external_work_dt = 0.0, 0.0, 0.0
    nb_inserted, nb_contact = 0, 0
    e_restitution = args.restitution  # restitution coefficient for nonsmooth contact
    edis_last_check = 0.0  # Energy value at the last checkpoint
    initial_energy = compute_total_energy(model, contact_dissipation)
    penalty = reference_stiffness if args.contact_type == "penalty" else None

    wall_active = False if args.box else True
    wall_check_active = False
    conv_streak_counter = 0
    required_streak = 3

    # --- Main Simulation Loop ---
    while t_current < max_time:

        # Boundary conditions
        if args.apply_bc:
            apply_bc(model, v_edge * dt)

        # Predictor / Residual
        model.beforeSolveStep()
        v = model.getVelocity().flatten()
        predictor(model, dt)

        if args.box and wall_check_active:
            model.getExternalForce()[:] = 0.0
            if args.contact_type == "penalty":
                contact_detected = apply_wall_contact_forces(
                    model, mesh, wall_gap=wall_gap, penalty=penalty, side="both"
                )
                if contact_detected and not wall_active:
                    logger.info("Contact with rigid walls detected")
                    wall_active = True

        r = update_residual(model)

        # Solve and corrector
        if nb_inserted > 0 and args.contact_type == "nonsmooth":
            active = check_active_contacts(model, H)

            if args.box and wall_check_active:
                H_solve, active_solve, contact_detected = (  # noqa: N806
                    append_wall_constraints(  # noqa: N806
                        model, mesh, H, active, wall_gap=wall_gap, side="both"
                    )
                )
                if contact_detected and not wall_active:
                    logger.info("Contact with rigid walls detected")
                    wall_active = True
            else:
                H_solve, active_solve = H, active  # noqa: N806

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
        if external_work_dt < 0 and args.apply_bc:
            model.getBlockedDOFs()[:] = False
            args.apply_bc = False
            initial_energy = compute_total_energy(model, contact_dissipation)
            logger.info("Releasing boundary conditions")
            if args.box:
                wall_check_active = True

        # Cohesive update
        inserted = model.checkCohesiveStress()
        if inserted > 0:
            H, m_inv, free = cohesive_update(  # noqa: N806
                model, mesh, dof_manager, build_H=(args.contact_type == "nonsmooth")
            )
            nb_inserted += inserted

        # Convergence Checks
        if (
            step > 0
            and step % conv_check_steps == 0
            and nb_inserted > 0
            and wall_active
        ):
            edis = model.getEnergy("dissipated")
            edis += contact_dissipation

            # Compare current energy vs energy 1 transit ago
            denom = max(abs(edis_last_check), 1e-20)
            rel_change = abs(edis - edis_last_check) / denom
            avg_active = nb_contact / (step + 1)
            logger.info(
                f"Time: {t_current:.2e}s | Inserted: {nb_inserted} | Avg active: {avg_active:.2f} | dE_dis: {rel_change:.2e}"
            )

            if rel_change < conv_tol:
                conv_streak_counter += 1
                logger.info(
                    f"Check passed ({conv_streak_counter}/{required_streak}). dE: {rel_change:.2e}"
                )
            else:
                conv_streak_counter = 0

            if conv_streak_counter >= required_streak:
                logger.info(
                    f"Global Convergence Reached!\n"
                    f"  - Energy stable ({rel_change:.1e}).\n"
                    f"  - Stopping simulation."
                )
                break

            current_energy = compute_total_energy(model, contact_dissipation)
            if current_energy > initial_energy * 3.0 or np.isnan(edis):
                logger.warning(
                    f"Instability Detected: E={current_energy:.2e} J > 3 * E0={initial_energy:.2e} J\n"
                    f"  - Stopping simulation."
                )
                break

            # Update checkpoint for the next interval
            edis_last_check = edis

        # --- Output ---

        # Paraview dump
        if step % (int(dump_freq * 2)) == 0:
            model.dump()

        # HDF5 dump
        if step % (int(dump_freq / 4)) == 0:
            avg_active = nb_contact / (step + 1)
            logger.info(
                f"Time: {t_current:.2e}s | Inserted: {nb_inserted} | Avg active: {avg_active:.2f}"
            )

            dump_step_h5(
                model,
                mesh,
                step,
                dt,
                contact_dissipation,
                h5_path,
                external_work,
                save_stress=True,
            )

        # Update time and step
        t_current += dt
        step += 1

    # --- Finalize ----
    elapsed = time.time() - start_time
    logger.info(f"Simulation finished. Elapsed time: {elapsed:.2f} s")

    with h5py.File(h5_path, "a", libver="latest") as f:
        f.attrs["simulation_time"] = elapsed
        f.attrs["contact_type"] = args.contact_type


if __name__ == "__main__":

    args = parse_simulation_args()
    run(args)
