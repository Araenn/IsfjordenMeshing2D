"""Isfjorden main: generate one conforming all-quad .msh, with automatic local repair."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator


# Edit VS_MODEL_PATH once, then run: python isfjorden_msh.py
ROOT = Path(__file__).resolve().parent
VS_MODEL_PATH = Path(
    "/home/lea/Desktop/code/resonance_model/resonance_model/picking/"
    "vs_model_2d_merged.csv"
)
GEOMETRY_PATH = ROOT / "data" / "Isfjorden.txt"
OUTPUT_DIR = ROOT / "output_mesh"

# Original physical and mesh parameters.
FMAX = 50.0
NPPW = 6
MIN_VS = 100.0
VS_BIN = 10
N_VERTICAL = 20
LC_MIN = 1.0
LC_MAX = 20.0
BASEMENT_PADDING = 50.0


# Water-only mesh optimization. Geometry and material interfaces are unchanged.
WATER_LC_FAR = 8.0
WATER_BOTTOM_KEEP_DISTANCE = 5.0
WATER_BOTTOM_TRANSITION_DISTANCE = 35.0

# Keep the original 5 m water resolution around source/receiver neighbourhoods.
WATER_PROTECTED_LC = 5.0
WATER_PROTECTED_KEEP_RADIUS = 20.0
WATER_PROTECTED_TRANSITION_RADIUS = 50.0

# Acoustic source and receiver-group centres. The 20 m protected radius fully
# contains the current +/-10 m ST receiver ellipses.
ACOUSTIC_SOURCE = (3290.0, 0.0)
ACOUSTIC_RECEIVER_CENTRES = (
    (4573.68, -120.98),   # STA
    (16674.96, -159.86),  # STB
)

# Keep the known-good full-quad route while validating adaptive water sizing.
# Set to False later to benchmark mesh generation without global subdivision.
USE_GLOBAL_QUAD_SUBDIVISION = False

# Fully structured quads in sediment cells. 95% is a controlled compromise:
# because all vertical curves in a layer must share their node count, one deep
# outlier must not force huge refinement over the full 20 km section.
# Set to 100.0 to enforce the original target size even in the deepest column.
SEDIMENT_VERTICAL_PERCENTILE = 95.0
MAX_SEDIMENT_QUADS = 3_500_000



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


def generate_mesh(x, basement, seafloor, interp_vs, output_dir, show_gui, outer_algorithm=5):
    """Build connected water, sediment and basement domains, then write outputs."""
    import gmsh
    import meshing_functions as msh

    lc_water = 1500.0 / (FMAX * NPPW)
    lc_basement = 1600.0 / (FMAX * NPPW)
    lc_sed = msh.lc_from_vs(
        interp_vs(x, 0.5 * (seafloor + basement)), FMAX, NPPW, LC_MIN, LC_MAX
    )
    # If the requested structured grid would explode the element count,
    # abort before allocating 100k surfaces and running the 2-D mesher.
    msh.configure_transfinite_sediment_mesh(
        {}, x, seafloor, basement, lc_sed,
        n_vertical=N_VERTICAL,
        vertical_percentile=SEDIMENT_VERTICAL_PERCENTILE,
        max_sediment_quads=MAX_SEDIMENT_QUADS,
        dry_run=True,
    )
    # Preflight output safety before any expensive Gmsh generation.
    if (output_dir / 'Isfjorden.msh').exists() or (output_dir / 'materials_generated.json').exists():
        raise RuntimeError(f'{output_dir} already contains a canonical mesh/materials. '
                           'Move/backup the previous result explicitly before generating again.')
    # Do not regenerate millions of cells if a previous repair failed: resume
    # from its raw Gmsh checkpoint through the existing repair_mesh.py tool.
    checkpoint = output_dir / 'Isfjorden.unrepaired.msh'
    if checkpoint.exists():
        raise RuntimeError(f'Unrepaired mesh checkpoint already exists: {checkpoint}. '
                           'Resume with repair_mesh.py --repair instead of generating again.')
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("isfjorden")
        # Shared interface points and curves give conforming domain boundaries.
        top_pts = msh.add_points(seafloor, np.minimum(lc_sed, lc_water), x)
        bottom_pts = msh.add_points(basement, np.minimum(lc_sed, lc_basement), x)
        top_edges = msh.add_edges(top_pts)
        bottom_edges = msh.add_edges(bottom_pts)
        water, water_boundary = msh.add_outer_domain(
            top_pts, top_edges, x, 0.0, lc_water, above=True
        )
        (
            sediment_materials, sediment_groups,
            interface_groups, sediment_boundary, structured_grid,
        ) = msh.add_layer_variable_vs_2d(
            seafloor, basement, x, interp_vs, lc_sed,
            top_pts, bottom_pts, top_edges, bottom_edges,
            n_vertical=N_VERTICAL, vs_bin=VS_BIN, min_vs=MIN_VS, f_att=FMAX,
        )
        rock, rock_boundary = msh.add_outer_domain(
            bottom_pts, bottom_edges, x, float(basement.min() - BASEMENT_PADDING),
            lc_basement, above=False,
        )
        gmsh.model.geo.synchronize()

        msh.configure_transfinite_sediment_mesh(
            structured_grid, x, seafloor, basement, lc_sed,
            n_vertical=N_VERTICAL,
            vertical_percentile=SEDIMENT_VERTICAL_PERCENTILE,
            max_sediment_quads=MAX_SEDIMENT_QUADS,
        )

        interface_lc = np.minimum(lc_sed, lc_water)
        protected_points = (ACOUSTIC_SOURCE, *ACOUSTIC_RECEIVER_CENTRES)
        msh.configure_adaptive_water_mesh(
            water_surface=water,
            x=x,
            seafloor=seafloor,
            interface_lc=interface_lc,
            lc_far=WATER_LC_FAR,
            bottom_keep_distance=WATER_BOTTOM_KEEP_DISTANCE,
            bottom_transition_distance=WATER_BOTTOM_TRANSITION_DISTANCE,
            protected_points=protected_points,
            protected_lc=WATER_PROTECTED_LC,
            protected_keep_radius=WATER_PROTECTED_KEEP_RADIUS,
            protected_transition_radius=WATER_PROTECTED_TRANSITION_RADIUS,
        )

        # 2-D material blocks.
        msh.add_physical_group([water], 1, "socle_1")
        msh.add_physical_group([rock], 3, "socle_3")
        for index, (name, surfaces) in enumerate(sediment_groups.items()):
            msh.add_physical_group(surfaces, 1000 + index, name)

        # 1-D side sets are defined explicitly in the geometry.  This avoids
        # inferring boundaries from node coordinates in gmsh2exodus.py.
        msh.add_physical_curve_group(
            [water_boundary["left"], *sediment_boundary["left"], rock_boundary["left"]],
            20000, "xmin",
        )
        msh.add_physical_curve_group(
            [water_boundary["right"], *sediment_boundary["right"], rock_boundary["right"]],
            20001, "xmax",
        )
        msh.add_physical_curve_group([rock_boundary["outer"]], 20002, "zmin")
        msh.add_physical_curve_group([water_boundary["outer"]], 20003, "zmax")

        # Only the upper sediment row is coupled to water.  Keep the material
        # name in the side-set name so SPECFEM-WAVE sees the actual two blocks.
        for index, (name, curves) in enumerate(interface_groups.items()):
            msh.add_physical_curve_group(
                curves, 21000 + index, f"socle_1|{name}"
            )

        mesh_path = output_dir / "Isfjorden.msh"
        if mesh_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing mesh: {mesh_path}")
        msh.final_meshing(
            mesh_path,
            use_global_quad_subdivision=USE_GLOBAL_QUAD_SUBDIVISION,
            water_surface=water,
            sediment_surfaces=[
                surface
                for surfaces in sediment_groups.values()
                for surface in surfaces
            ],
            rock_surface=rock,
            top_interface_curves=top_edges,
            bottom_interface_curves=bottom_edges,
            unstructured_algorithm=outer_algorithm,
        )
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
        if materials_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing material file: {materials_path}")
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
    parser.add_argument("--outer-algorithm", type=int, choices=(5, 6, 8), default=5,
                        help="Classical unstructured 2-D algorithm for water/basement: "
                             "5=Delaunay (default); 6=Frontal-Delaunay; 8=Frontal-quad.")
    parser.add_argument("--check-inputs", action="store_true", help="Validate inputs without meshing.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Estimate structured sediment element count without building geometry.")
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(resolve_path(args.geometry), resolve_path(args.vs_model))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.check_inputs:
        print("Inputs are valid.")
        return
    if args.plan_only:
        import meshing_functions as msh
        x, basement, seafloor, interp_vs = inputs
        lc_sed = msh.lc_from_vs(
            interp_vs(x, 0.5 * (seafloor + basement)),
            FMAX, NPPW, LC_MIN, LC_MAX,
        )
        msh.configure_transfinite_sediment_mesh(
            {}, x, seafloor, basement, lc_sed,
            n_vertical=N_VERTICAL,
            vertical_percentile=SEDIMENT_VERTICAL_PERCENTILE,
            max_sediment_quads=MAX_SEDIMENT_QUADS,
            dry_run=True,
        )
        return
    generate_mesh(*inputs, resolve_path(args.output_dir), not args.no_gui,
                  outer_algorithm=args.outer_algorithm)


if __name__ == "__main__":
    main()
