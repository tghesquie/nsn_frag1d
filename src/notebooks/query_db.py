"""Minimal query layer for nsn-frag1d simulation outputs.

Examples
--------
>>> from query_db import query, load, scan, studies
>>>
>>> # List available studies
>>> studies()

>>> # Scan a study for run metadata
>>> df = scan("strain_rate_nsn_nobox_study")
>>>
>>> # Query by parameters (single value)
>>> df = query(study="strain_rate_nsn_nobox_study", r=100.0, seed=1)
>>>
>>> # Query by parameters (list of values)
>>> df = query(study="strain_rate_nsn_nobox_study", r=[1.0, 10.0, 100.0], seed=1)
>>>
    >>> # Query by relative path
    >>> df = query(path="data/strain_rate_nsn_nobox_study/nsnfrag1d_l1.00e-01_.../data.h5")
    >>>
    >>> # Load one file directly
>>> ts = load("path/to/data.h5")
>>> # Metadata only
>>> df = query(study="strain_rate_nsn_nobox_study", r=100.0, load_data=False)
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------
_SCHEMA: dict[str, type] = {
    "l": float,
    "n": float,
    "md": float,
    "p": int,
    "t": float,
    "r": float,
    "s": float,
    "dd": float,
    "k": float,
    "e": float,
    "sc": float,
    "iv": float,
    "bf": float,
    "cir": float,
}

_FLAGS = {"bc": "apply_bc", "box": "box"}
"""Boolean flags decoded from the run ID (``bc`` → ``apply_bc``, ``box`` → ``box``)."""

# Per-study defaults for parameters not encoded in the run ID.
# These values are used by scan() to fill missing columns so that notebooks
# and queries can rely on a consistent schema across studies.
_STUDY_DEFAULTS: dict[str, dict[str, Any]] = {
    "strain_rate_nsn_nobox_study": {
        "contact_type": "nonsmooth",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": False,
    },
    "strain_rate_pen_nobox_study": {
        "contact_type": "penalty",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": False,
    },
    "timestep_nobox_study": {
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": False,
    },
    "impact_study": {
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.5,
        "apply_bc": True,
        "box": True,
    },
    "box_size_study": {
        "contact_type": "nonsmooth",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": True,
    },
    "restitution_coefficient_study": {
        "contact_type": "nonsmooth",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": True,
    },
    "bar_length_box_study": {
        "contact_type": "nonsmooth",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": True,
    },
    "stress_heatmap_study": {
        "contact_type": "nonsmooth",
        "mesh_variation": 0.4,
        "n_dumps": 1000,
        "cohesive_insertion_ratio": 0.0,
        "apply_bc": True,
        "box": True,
    },
}

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _resolve_study(study: str | Path) -> Path:
    """Resolve a study name or path to a directory."""
    study_path = Path(study)
    if study_path.is_dir():
        return study_path.resolve()

    # Try common relative paths, e.g. ``data/<study>`` from project root.
    candidates = [_DATA_ROOT / study]
    if isinstance(study, str) and study.startswith("data/"):
        candidates.append(_DATA_ROOT / study[5:])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(f"Study not found: {study!r}")


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Parse an ``nsnfrag1d_...`` run identifier into typed parameters.

    The run ID encodes geometry, mesh, time, loading, contact, and boundary-
    condition parameters. Boolean flags such as ``bc`` and ``box`` are inferred
    from the presence of the corresponding token. The contact type (penalty vs.
    nonsmooth) is inferred from the presence of ``k`` (penalty factor) or
    ``e``/``sc`` (restitution/cohesive factor).

    Parameters
    ----------
    run_id:
        Run directory name, e.g. ``"nsnfrag1d_l1.00e-01_n5.00e+04_md5.00e+05_..."``.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter names to parsed values.
    """
    parts = run_id.split("_")
    if not parts or parts[0] != "nsnfrag1d":
        raise ValueError(f"Run ID must start with 'nsnfrag1d', got: {run_id!r}")

    params: dict[str, Any] = {}
    for token in parts[1:]:
        if token in _FLAGS:
            params[_FLAGS[token]] = True
            continue
        if token.startswith("seed"):
            params["seed"] = int(token[4:])
            continue

        match = re.match(r"^([a-z]+)(.*)$", token)
        if not match:
            continue
        key, raw = match.groups()

        if key == "sc":
            raw = raw.lstrip("-")
            params["sc"] = "auto" if raw == "auto" else float(raw)
        elif key in _SCHEMA:
            params[key] = _SCHEMA[key](raw)

    if "k" in params:
        params["contact_type"] = "penalty"
    elif "e" in params or "sc" in params:
        params["contact_type"] = "nonsmooth"

    return params




def _locate_h5(path: Path) -> Path:
    """Resolve a path to a single HDF5 file.

    Supports:
      - direct file path
      - new layout: ``<run>/data/data.h5``
      - legacy layout: ``<run>/*.h5``
    """
    if path.is_file():
        return path

    new_layout = path / "data" / "data.h5"
    if new_layout.exists():
        return new_layout

    candidates = list(path.glob("*.h5"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No HDF5 file found in {path}")
    raise FileNotFoundError(
        f"Expected one HDF5 file in {path}, found {len(candidates)}"
    )


def _run_dir_from_data_path(data_path: Path) -> Path:
    """Return the run directory given a data.h5 path.

    New layout: ``<run>/data/data.h5`` -> ``<run>``.
    Legacy layout: ``<run>/data.h5``    -> ``<run>``.
    """
    if data_path.parent.name == "data":
        return data_path.parent.parent
    return data_path.parent

@lru_cache(maxsize=32)
def scan(study: str | Path, pattern: str = "data.h5") -> pd.DataFrame:
    """Return a metadata DataFrame of all runs found in ``study``."""
    root = _resolve_study(study)
    records: list[dict[str, Any]] = []

    for data_path in root.rglob(pattern):
        data_path = data_path.resolve()
        run_dir = _run_dir_from_data_path(data_path)
        run_id = run_dir.name

        try:
            params = parse_run_id(run_id)
        except ValueError:
            continue

        record = {
            "study_name": root.name,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "data_path": str(data_path),
            **params,
        }

        defaults = _STUDY_DEFAULTS.get(root.name, {})
        for key, value in defaults.items():
            record.setdefault(key, value)

        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    leading = ["study_name", "run_id", "run_dir", "data_path"]
    param_cols = [
        "l", "n", "md", "p", "t", "r", "s", "dd", "k", "e", "sc",
        "iv", "bf", "cir", "seed", "contact_type", "apply_bc", "box",
        "mesh_variation", "n_dumps", "cohesive_insertion_ratio",
    ]
    ordered = leading + [c for c in param_cols if c in df.columns]
    ordered += [c for c in sorted(df.columns) if c not in ordered]
    return df[ordered].copy()


def _match(col: pd.Series, value: Any, tol: float = 1e-12) -> pd.Series:
    """Match a column against a single value or list of values."""
    if isinstance(value, (list, tuple, np.ndarray)):
        masks = [_match(col, v, tol) for v in value]
        return pd.Series(np.any(masks, axis=0), index=col.index)

    if pd.api.types.is_float_dtype(col) and isinstance(
        value, (int, float, np.floating, np.integer)
    ):
        return pd.Series(
            np.isclose(col.to_numpy(), float(value), atol=tol, rtol=0.0),
            index=col.index,
        )
    return col == value


def _filter(df: pd.DataFrame, tol: float = 1e-12, **params: Any) -> pd.DataFrame:
    """Filter a runs DataFrame, supporting list values."""
    if df.empty or not params:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for key, value in params.items():
        if key not in df.columns:
            raise KeyError(
                f"Parameter {key!r} not found. Available: {df.columns.tolist()}"
            )
        mask &= _match(df[key], value, tol)

    return df[mask].copy()


def find_runs(study: str | Path, **filters: Any) -> pd.DataFrame:
    """Metadata-only alias for :func:`query` (backward-compatible)."""
    return query(study=study, load_data=False, **filters)


def filter_runs(df: pd.DataFrame, tol: float = 1e-12, **filters: Any) -> pd.DataFrame:
    """Filter a runs DataFrame, supporting list values."""
    return _filter(df, tol=tol, **filters)


def load(path: str | Path) -> pd.DataFrame:
    """Load a ``data.h5`` file into a time-series DataFrame."""
    path = _locate_h5(Path(path))

    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        sim_time = f.attrs.get("simulation_time", 0.0)
        contact_type = f.attrs.get("contact_type", None)

        for grp_name, grp in f.items():
            if not grp_name.startswith("step_"):
                continue
            row: dict[str, Any] = {"step": int(grp_name.split("_")[1])}
            row.update(dict(grp.attrs))
            row.update({k: v[()] for k, v in grp.items()})
            rows.append(row)

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        df = pd.DataFrame(columns=["step"])

    df["simulation_time"] = sim_time
    if contact_type is not None:
        df["contact_type_h5"] = contact_type

    if "step" in df.columns:
        df = df.set_index("step").sort_index()

    return df


load_run_h5 = load


def _query_path(path: str | Path, load_data: bool = True) -> pd.DataFrame:
    """Build a one-row DataFrame from a relative path."""
    data_path = _locate_h5(Path(path))
    run_dir = _run_dir_from_data_path(data_path)
    run_id = run_dir.name
    params = parse_run_id(run_id)

    record = {
        "study_name": run_dir.parent.name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        **params,
    }

    df = pd.DataFrame([record])
    if load_data:
        df["data"] = [load(data_path)]
    return df


def query(
    study: str | Path | None = None,
    path: str | Path | None = None,
    load_data: bool = True,
    **params: Any,
) -> pd.DataFrame:
    """Find simulation runs and return them ready to plot.

    Parameters
    ----------
    study:
        Study name (resolved under ``data/data/``) or path to a study root.
    path:
        Relative path to a run directory or ``data.h5`` file.
    load_data:
        If True (default), attach a ``data`` column with the loaded time series
        for each run.
    **params:
        Parameter filters. Values can be single values or lists.

    Returns
    -------
    pd.DataFrame
        One row per matching run. Columns include metadata, parsed parameters,
        ``data_path``, and optionally ``data``.

    Examples
    --------
    >>> from query_db import query
    >>>
    >>> # All runs of a study
    >>> df = query(study="strain_rate_nsn_nobox_study")
    >>>
    >>> # Filter by parameter
    >>> df = query(study="strain_rate_nsn_nobox_study", r=100.0, seed=1)
    >>>
    >>> # List of parameter values
    >>> df = query(study="strain_rate_nsn_nobox_study", r=[1.0, 10.0, 100.0])
    >>>
    >>> # Load a fresh README example run from output/
    >>> df = query(path="../output/test_penalty/data.h5")
    >>>
    >>> # List every run under output/
    >>> df = query(study="../output", load_data=False)
    """
    if path is not None and study is not None:
        raise ValueError("Provide either ``study`` or ``path``, not both.")

    if path is not None:
        return _query_path(path, load_data=load_data)

    if study is None:
        raise ValueError("Provide either ``study`` or ``path``.")

    df = _filter(scan(study), **params)
    if df.empty:
        raise ValueError(f"No run found matching {params!r}")

    if load_data:
        df["data"] = [load(p) for p in df["data_path"]]

    return df.reset_index(drop=True)


def studies(root: str | Path = _DATA_ROOT) -> list[str]:
    """List available study names under ``root``."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted([d.name for d in root_path.iterdir() if d.is_dir()])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query nsn-frag1d simulation outputs."
    )
    parser.add_argument(
        "study",
        nargs="?",
        default=None,
        help="Study name or directory to scan.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Relative path to a run directory or data.h5 file.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Parameter filter (can be given multiple times).",
    )
    parser.add_argument(
        "--no-load",
        dest="load_data",
        action="store_false",
        default=True,
        help="Return metadata only, do not load HDF5 data.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output CSV (excludes loaded data columns).",
    )
    args = parser.parse_args(argv)

    if args.path:
        df = query(path=args.path, load_data=args.load_data)
    elif args.study:
        filters: dict[str, Any] = {}
        for item in args.filter:
            if "=" not in item:
                parser.error(f"Filter must be KEY=VALUE, got: {item!r}")
            key, value = item.split("=", 1)
            filters[key] = _coerce(value)
        df = query(args.study, load_data=args.load_data, **filters)
    else:
        parser.error("Provide either a STUDY or --path")

    if args.csv:
        print(df.drop(columns=[c for c in ["data"] if c in df.columns]).to_csv(index=False))
    else:
        print(df.drop(columns=[c for c in ["data"] if c in df.columns]).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
