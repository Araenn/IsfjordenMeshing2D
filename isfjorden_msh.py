"""Generate an Isfjorden mesh from geometry and an external 2D Vs CSV."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator


# Edit VS_MODEL_PATH once, then run: python isfjordenmsh.py
ROOT = Path(__file__).resolve().parent
VS_MODEL_PATH = Path(
    "/home/lea/Desktop/code/resonance_model/resonance_model/picking/"
    "vs_model_2d_merged.csv"
)
GEOMETRY_PATH = ROOT / "data" / "Isfjorden.txt"
OUTPUT_DIR = ROOT / "output"

# Original physical and mesh parameters.
FMAX = 50.0
NPPW = 6
MIN_VS = 100.0
VS_BIN = 10
N_VERTICAL = 20
LC_MIN = 1.0
LC_MAX = 20.0
BASEMENT_PADDING = 50.0


def resolve_path(path):
    """Resolve relative configuration paths from the repository directory."""
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_inputs(geometry_path, vs_model_path):
    """Validate inputs and build the original nearest-neighbour Vs interpolator."""
    if not geometry_path.is_file():
        raise ValueError(f"Geometry file not found: {geometry_path}")
    if not vs_model_path.is_file():
        raise ValueError(
            f"External Vs model not found: {vs_model_path}\n"
            "Set VS_MODEL_PATH, ISFJORDEN_VS_MODEL, or --vs-model. "
            "Expected CSV columns: x_km, z_m, vs_mps."
        )
    geometry = np.loadtxt(geometry_path, skiprows=1, ndmin=2)
    if geometry.shape[0] < 2 or geometry.shape[1] != 3:
        raise ValueError("Geometry must contain at least two rows of Distance_m, Basement, Seafloor.")
    if not np.isfinite(geometry).all():
        raise ValueError("Geometry contains non-finite values.")
    x, basement, seafloor = geometry.T
    x = x - x[0]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Geometry distances must be strictly increasing.")
    if np.any(basement >= seafloor) or np.any(seafloor >= 0):
        raise ValueError("Geometry requires Basement < Seafloor < 0 (elevation in metres).")

    model = pd.read_csv(vs_model_path)
    required = ["x_km", "z_m", "vs_mps"]
    if not set(required).issubset(model.columns):
        raise ValueError(f"Vs CSV must contain columns: {', '.join(required)}")
    values = model[required].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Vs CSV must contain finite numeric samples.")
    if np.any(values[:, 2] <= 0):
        raise ValueError("Vs values must be strictly positive.")
    points = np.column_stack((values[:, 0] * 1000.0, values[:, 1]))
    interp = NearestNDInterpolator(points, np.maximum(values[:, 2], MIN_VS))
    print(f"Geometry: {len(x)} columns, {x[-1] / 1000:.3f} km")
    print(f"External Vs model: {vs_model_path} ({len(values)} samples)")
    if x[0] < points[:, 0].min() or x[-1] > points[:, 0].max():
        print("Warning: geometry extends beyond the CSV x range; nearest values are extrapolated.")
    return x, basement, seafloor, interp


def generate_mesh(x, basement, seafloor, interp_vs, output_dir, show_gui):
    """Build connected water, sediment and basement domains, then write outputs."""
    import gmsh
    import meshing_functions as msh

    lc_water = 1500.0 / (FMAX * NPPW)
    lc_basement = 1600.0 / (FMAX * NPPW)
    lc_sed = msh.lc_from_vs(
        interp_vs(x, 0.5 * (seafloor + basement)), FMAX, NPPW, LC_MIN, LC_MAX
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("isfjorden")
        # Shared interface points and curves give conforming domain boundaries.
        top_pts = msh.add_points(seafloor, np.minimum(lc_sed, lc_water), x)
        bottom_pts = msh.add_points(basement, np.minimum(lc_sed, lc_basement), x)
        top_edges = msh.add_edges(top_pts)
        bottom_edges = msh.add_edges(bottom_pts)
        water = msh.add_outer_domain(
            top_pts, top_edges, x, 0.0, lc_water, above=True
        )
        sediment_materials, sediment_groups = msh.add_layer_variable_vs_2d(
            seafloor, basement, x, interp_vs, lc_sed,
            top_pts, bottom_pts, top_edges, bottom_edges,
            n_vertical=N_VERTICAL, vs_bin=VS_BIN, min_vs=MIN_VS, f_att=FMAX,
        )
        rock = msh.add_outer_domain(
            bottom_pts, bottom_edges, x, float(basement.min() - BASEMENT_PADDING),
            lc_basement, above=False,
        )
        gmsh.model.geo.synchronize()
        msh.add_physical_group([water], 1, "socle_1")
        msh.add_physical_group([rock], 3, "socle_3")
        for index, (name, surfaces) in enumerate(sediment_groups.items()):
            msh.add_physical_group(surfaces, 1000 + index, name)

        mesh_path = output_dir / "Isfjorden.msh"
        msh.final_meshing(mesh_path)
        materials = {
            "acoustic_2d": {"socle_1": {"rho": 1000, "vp": 1500}},
            "viscoelastic_2d": {
                **sediment_materials,
                "socle_3": {
                    "rho": 2400, "vp": 4000, "vs": 1600,
                    "q_kappa": 250, "q_mu": 150, "f_att": FMAX,
                },
            },
        }
        materials_path = output_dir / "materials_generated.json"
        materials_path.write_text(json.dumps(materials, indent=2) + "\n", encoding="utf-8")
        print(f"Mesh written: {mesh_path}")
        print(f"Materials written: {materials_path}")
        print(f"Sediment materials: {len(sediment_materials)}")
        if show_gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vs-model", type=Path,
        default=os.environ.get("ISFJORDEN_VS_MODEL", VS_MODEL_PATH),
        help="External CSV path (overrides environment variable and VS_MODEL_PATH).",
    )
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-gui", action="store_true", help="Generate without opening Gmsh.")
    parser.add_argument("--check-inputs", action="store_true", help="Validate inputs without meshing.")
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(resolve_path(args.geometry), resolve_path(args.vs_model))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.check_inputs:
        print("Inputs are valid.")
        return
    generate_mesh(*inputs, resolve_path(args.output_dir), not args.no_gui)


if __name__ == "__main__":
    main()
