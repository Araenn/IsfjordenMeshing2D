"""Read an external model pickle and export the geometry used for meshing."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_pickle_df(pickle_file):
    """Return model samples stored as a DataFrame or a dictionary of columns."""
    with Path(pickle_file).expanduser().open("rb") as stream:
        obj = pickle.load(stream)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    raise TypeError(
        f"Unsupported pickle content: {type(obj).__name__}. "
        "Expected a DataFrame or a dictionary of sample columns."
    )


def export_bathy_from_pkl(pickle_file, output_file, dx=4.08):
    """Export Distance_m, Basement and Seafloor on a regular horizontal grid.

    Input samples must contain x_km and z_m, using negative elevations below
    the sea surface. At each x, the largest z defines the seafloor and the
    smallest z defines the basement. The model must include both interfaces.
    As in the original exporter, the final input x position is excluded from
    the regular grid created with np.arange().
    """
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be a finite, strictly positive distance in metres.")
    df = load_pickle_df(pickle_file)
    required = ["x_km", "z_m"]
    if not set(required).issubset(df.columns):
        raise ValueError("The model pickle must contain x_km and z_m sample columns.")
    coordinates = df[required].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("The model must contain finite numeric x_km and z_m samples.")

    samples = pd.DataFrame({
        "x_m": coordinates[:, 0] * 1000.0,
        "z_m": coordinates[:, 1],
    })
    bathy = samples.groupby("x_m", sort=True)["z_m"].agg(
        seafloor="max", basement="min"
    ).reset_index()
    if len(bathy) < 2:
        raise ValueError("At least two distinct horizontal positions are required.")
    if (bathy["basement"] >= bathy["seafloor"]).any():
        raise ValueError("Each horizontal position must include a nonzero sediment thickness.")
    if (bathy["seafloor"] >= 0).any():
        raise ValueError("Seafloor elevations must be negative, as required by the meshing script.")

    x_input = bathy["x_m"].to_numpy()
    x_new = np.arange(x_input[0], x_input[-1], dx)
    if len(x_new) < 2:
        raise ValueError("dx is too large for this section: at least two output rows are required.")
    out = pd.DataFrame({
        "Distance_m": x_new - x_new[0],
        "Basement": np.interp(x_new, x_input, bathy["basement"]),
        "Seafloor": np.interp(x_new, x_input, bathy["seafloor"]),
    })
    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, sep="\t", index=False, float_format="%.6f")
    print(f"Geometry written: {output_file}")
    print(f"Input section origin: {x_input[0]:.6f} m; output starts at 0 m")
    print(f"Horizontal step: {dx:g} m; samples: {len(out)}")
    print(f"Maximum output distance: {out['Distance_m'].iloc[-1]:.6f} m")
    return out
