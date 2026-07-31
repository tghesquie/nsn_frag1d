# Copyright (c) 2026 EPFL
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting helpers for nsn-frag1d simulation outputs.

This module is meant to be used together with ``query_db``::

    import query_db
    import plots

    runs = query_db.query(study="data/restitution_coefficient_study", load_data=False)
    data = plots.load_run(runs.iloc[0])

    fig = plots.plot_energy_balance(data)
    fig.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import query_db

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_COLORSCALE = "Spectral_r"
_SAMPLES = [0.0, 0.1, 0.25, 0.35]

_COLORS = sample_colorscale(_COLORSCALE, _SAMPLES)
_COLORS.append("lightgrey")
_COLORS.append("grey")

_FILLCOLORS = [c.replace("rgb(", "rgba(").replace(")", ",0.7)") for c in _COLORS]

_ENERGY_LABELS = {
    "kinetic_energy": "ΔK",
    "potential_energy": "ΔU",
    "reversible_energy": "ΔR",
    "contact_energy": "ΔC_rev",
    "dissipated_energy": "ΔG",
    "contact_dissipation": "ΔC_dis",
}

_STACKED_LABELS = {
    "kinetic_energy": "K",
    "potential_energy": "U",
    "reversible_energy": "R",
    "contact_energy": "C_rev",
    "dissipated_energy": "G",
    "contact_dissipation": "C_dis",
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def get_layout() -> go.Layout:
    """Return the default Plotly layout used across all figures."""
    return go.Layout(
        xaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
            ticks="inside",
            exponentformat="power",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
            ticks="inside",
            exponentformat="power",
            tickfont=dict(size=12),
        ),
        font=dict(family="Latin-Modern", size=12, color="Black"),
        legend=dict(
            x=0.97,
            y=1.2,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            orientation="h",
            xanchor="right",
            yanchor="top",
        ),
        width=800,
        height=500,
        showlegend=True,
        template="plotly_white",
    )


def axis_style() -> dict[str, Any]:
    """Return the default axis style dictionary."""
    return dict(
        showgrid=True,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        zeroline=False,
        ticks="inside",
        exponentformat="power",
        tickfont=dict(size=12),
    )


def _update_figure(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    legend_y: float = 1.2,
) -> go.Figure:
    """Apply the default layout to a figure."""
    fig.update_layout(
        get_layout(),
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(y=legend_y),
    )
    return fig


def _output_path(
    data: dict[str, Any], filename: str, output_dir: str | None
) -> Path | None:
    """Build an output path when writing is requested."""
    if output_dir is None:
        output_dir = Path("../../output") / data["run_id"]
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run(path_or_row: str | Path | pd.Series) -> dict[str, Any]:
    """Load a single run into the dictionary expected by the plot functions.

    Parameters
    ----------
    path_or_row:
        Either a path to an HDF5 file/run directory, or a row from a
        ``query_db.query(..., load_data=False)`` DataFrame.

    Returns
    -------
    dict
        Dictionary with keys ``df``, ``run_id``, ``run_dir``, ``stress_times``,
        ``stress_matrix``, ``quad_coords``, ``nodes_coords`` and ``global_data``.
    """
    if isinstance(path_or_row, pd.Series):
        path = Path(path_or_row["data_path"])
        run_id = str(path_or_row["run_id"])
    else:
        path = Path(path_or_row)
        run_id = path.parent.name

    df = query_db.load(path)

    stress: list[tuple[int, float | None, np.ndarray]] = []
    global_data: dict[str, Any] = {}
    quad_coords: np.ndarray | None = None
    nodes_coords: np.ndarray | None = None

    with h5py.File(path, "r") as f:
        quad_coords = (
            np.asarray(f["quad_coordinates"][()]).flatten()
            if "quad_coordinates" in f
            else None
        )
        nodes_coords = (
            np.asarray(f["nodes_coordinates"][()]).flatten()
            if "nodes_coordinates" in f
            else None
        )

        for name, item in f.items():
            if isinstance(item, h5py.Group) and name.startswith("step_"):
                step_idx = int(name.split("_")[1])
                time_value = None
                if step_idx in df.index and "time" in df.columns:
                    time_value = float(df.loc[step_idx, "time"])

                stress_array = None
                if "stress" in item:
                    stress_array = item["stress"][()]
                elif "stress" in item.attrs:
                    stress_array = item.attrs["stress"]

                if stress_array is not None:
                    stress.append((step_idx, time_value, stress_array))

            elif isinstance(item, h5py.Dataset) and name not in [
                "quad_coordinates",
                "nodes_coordinates",
            ]:
                global_data[name] = item[()]

    if stress:
        stress.sort(key=lambda x: x[0])
        stress_times = np.array([x[1] for x in stress])
        stress_matrix = np.vstack([x[2] for x in stress])
    else:
        stress_times = None
        stress_matrix = None

    return {
        "df": df,
        "run_id": run_id,
        "run_dir": str(path.parent),
        "stress_times": stress_times,
        "stress_matrix": stress_matrix,
        "quad_coords": quad_coords,
        "nodes_coords": nodes_coords,
        "global_data": global_data,
    }


# ---------------------------------------------------------------------------
# Energy plots
# ---------------------------------------------------------------------------
def plot_energy_balance(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
) -> go.Figure:
    """Plot algorithmic and mechanical energy balance versus time."""
    df = data["df"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["algorithmic_energy_balance"]
            - df["algorithmic_energy_balance"].iloc[0],
            mode="lines",
            name="ΔE_alg",
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["mechanical_energy_balance"] - df["mechanical_energy_balance"].iloc[0],
            mode="lines",
            name="ΔE_mech",
            line=dict(color="red"),
        )
    )

    _update_figure(
        fig,
        title="Energy balance vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="ΔE (J)",
        legend_y=0.97,
    )

    if write:
        out = _output_path(data, "energy_balance_vs_time_2.pdf", output_dir)
        fig.write_image(str(out), width=400, height=300, scale=2)

    return fig


def plot_energies(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
) -> go.Figure:
    """Plot individual energy variations versus time."""
    df = data["df"]

    energy_keys = [
        "kinetic_energy",
        "potential_energy",
        "reversible_energy",
        "contact_energy",
        "dissipated_energy",
        "contact_dissipation",
    ]

    fig = go.Figure()
    for key, color in zip(energy_keys, _COLORS):
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[key] - df[key].iloc[0],
                mode="lines",
                name=_ENERGY_LABELS[key],
                line=dict(color=color),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=-(df["external_work"] - df["external_work"].iloc[0]),
            mode="lines",
            name="W_ext",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["mechanical_energy_balance"] - df["mechanical_energy_balance"].iloc[0],
            mode="lines",
            name="ΔE_mech",
            line=dict(color="black"),
        )
    )

    _update_figure(
        fig,
        title="Energy variation vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="ΔE (J)",
        legend_y=1.2,
    )

    if write:
        out = _output_path(data, "energy_vs_time.pdf", output_dir)
        fig.write_image(str(out), width=400, height=400, scale=2)

    return fig


def plot_energies_stack(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
) -> go.Figure:
    """Plot stacked energy components versus time."""
    df = data["df"]

    ekin = df["kinetic_energy"]
    epot = ekin + df["potential_energy"]
    erev = epot + df["reversible_energy"]
    econrev = erev + df["contact_energy"]
    edis = econrev + df["dissipated_energy"]
    econdis = edis + df["contact_dissipation"]
    einj = (
        df["external_work"]
        + df["kinetic_energy"].iloc[0]
        + df["potential_energy"].iloc[0]
    )

    stacks = [ekin, epot, erev, econrev, edis, econdis]
    energy_keys = [
        "kinetic_energy",
        "potential_energy",
        "reversible_energy",
        "contact_energy",
        "dissipated_energy",
        "contact_dissipation",
    ]

    fig = go.Figure()
    for i, (stack, key, color, fillcolor) in enumerate(
        zip(stacks, energy_keys, _COLORS, _FILLCOLORS)
    ):
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=stack,
                mode="lines",
                name=_STACKED_LABELS[key],
                line=dict(color=color, width=0),
                fill="tozeroy" if i == 0 else "tonexty",
                fillcolor=fillcolor,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=einj,
            mode="lines",
            name="W_ext + E0",
            line=dict(color="red", width=2),
        )
    )

    _update_figure(
        fig,
        title="Stacked Energy vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="E (J)",
        legend_y=0.9,
    )

    if write:
        out = _output_path(data, "stacked_energy_vs_time.pdf", output_dir)
        fig.write_image(str(out), width=400, height=400, scale=2)

    return fig


# ---------------------------------------------------------------------------
# Fragment count
# ---------------------------------------------------------------------------
def plot_fragment_count(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
) -> go.Figure:
    """Plot number of fragments versus time."""
    df = data["df"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["nb_fragments"],
            mode="lines",
            name="Number of fragments",
            line=dict(color="black"),
        )
    )

    _update_figure(
        fig,
        title="Number of fragments vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="Number of fragments",
        legend_y=1.2,
    )

    if write:
        out = _output_path(data, "n_fragments_vs_time.pdf", output_dir)
        fig.write_image(str(out), width=400, height=400, scale=2)

    return fig


# ---------------------------------------------------------------------------
# Fragment mass distribution (CT scan)
# ---------------------------------------------------------------------------
def plot_fragment_mass_ct(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
    nbinsx: int = 100,
) -> go.Figure:
    """Plot normalized fragment mass distribution as a function of time."""
    df = data["df"]

    final_masses = np.asarray(df["fragment_mass"].iloc[-1])
    total_mass = float(np.sum(final_masses))
    final_masses_norm = final_masses / total_mass
    lmin, lmax = float(np.min(final_masses_norm)), float(np.max(final_masses_norm))
    bins = np.linspace(lmin, lmax, nbinsx + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    y_vals = df["time"].values if "time" in df.columns else df.index.values
    Z = np.empty((len(df), nbinsx), dtype=float)
    for i, masses in enumerate(df["fragment_mass"].values):
        masses_norm = masses / total_mass
        counts, _ = np.histogram(masses_norm, bins=bins)
        Z[i, :] = counts

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=bin_centers,
            y=y_vals,
            z=Z,
            colorscale="Greys",
            colorbar=dict(title="Number of fragments"),
        )
    )

    _update_figure(
        fig,
        title="Normalized Fragment Mass Distribution (CT scan)",
        xaxis_title="Mass / M̄",
        yaxis_title="Time",
        legend_y=1.2,
    )

    if write:
        out = _output_path(data, "fragment_mass_ct_scan_norm.pdf", output_dir)
        fig.write_image(str(out), width=400, height=400, scale=2)

    return fig


# ---------------------------------------------------------------------------
# Space-time stress diagram
# ---------------------------------------------------------------------------
def plot_stress_space_time(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
    downsample_factor: int = 1,
) -> go.Figure:
    """Plot the space-time stress diagram as a heatmap."""
    quad_coords = data.get("quad_coords")
    stress_times = data.get("stress_times")
    stress_matrix = data.get("stress_matrix")

    if quad_coords is None or stress_times is None or stress_matrix is None:
        raise ValueError(
            "Cannot plot stress diagram: missing quad_coords, stress_times or stress_matrix."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=quad_coords,
            y=stress_times[::downsample_factor],
            z=stress_matrix[::downsample_factor, :],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-3e8,
            zmax=3e8,
            colorbar=dict(title="Stress [Pa]", thickness=15),
        )
    )

    _update_figure(
        fig,
        title=f"Space-Time Stress Diagram ({data['run_id']})",
        xaxis_title="Position [m]",
        yaxis_title="Time [s]",
        legend_y=1.2,
    )
    fig.update_layout(yaxis=dict(autorange=True))

    if write:
        out = _output_path(data, f"{data['run_id']}_stress_xt_diagram.png", output_dir)
        fig.write_image(str(out), width=500, height=500, scale=4)

    return fig


# ---------------------------------------------------------------------------
# Joint fragment mass vs. velocity distribution
# ---------------------------------------------------------------------------
def plot_joint_mass_velocity(
    data: dict[str, Any],
    write: bool = False,
    output_dir: str | Path | None = None,
    step: int = -1,
) -> go.Figure:
    """Plot joint distribution of fragment mass and velocity."""
    df = data["df"]

    m_final = df["fragment_mass"].iloc[step]
    v_final = df["fragment_velocity"].iloc[step]

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
    )

    fig.add_trace(
        go.Scatter(
            x=m_final,
            y=v_final,
            mode="markers",
            marker=dict(
                color="black", opacity=0.4, size=4, line=dict(width=0.5, color="black")
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=m_final,
            nbinsx=100,
            marker_color="black",
            opacity=0.9,
            marker_line_width=0.0,
            marker_line_color="black",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            y=v_final,
            nbinsy=100,
            marker_color="black",
            opacity=0.9,
            marker_line_width=0.0,
            marker_line_color="black",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(get_layout())

    style = axis_style()
    fig.update_xaxes(style)
    fig.update_yaxes(style)
    fig.update_xaxes(title_text="Mass (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=2)

    fig.update_layout(
        title="Fragment Mass vs. Velocity Joint Distribution",
        width=600,
        height=600,
    )

    if write:
        out = _output_path(data, "joint_mass_velocity.pdf", output_dir)
        fig.write_image(str(out), scale=2)

    return fig
