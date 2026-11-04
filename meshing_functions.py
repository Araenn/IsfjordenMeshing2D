"""Isfjorden: all Gmsh geometry, meshing, local repair and validation functions."""

import gmsh
import numpy as np
import csv
import math
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

factory = gmsh.model.geo


def add_points(z_values, lc_values, x_values):
    lc_values = np.broadcast_to(lc_values, np.shape(x_values))
    return [
        factory.addPoint(float(x), float(z), 0, float(lc))
        for x, z, lc in zip(x_values, z_values, lc_values)
    ]


def add_edges(points):
    return [factory.addLine(a, b) for a, b in zip(points[:-1], points[1:])]


def add_outer_domain(interface_points, interface_edges, x, outer_z, lc, above):
    """Create water or basement and return its explicit exterior curves.

    The interface curves themselves are reused, so the mesh is conforming.
    Returning the exterior curves lets the caller create exact 1-D physical
    groups for xmin/xmax/zmin/zmax instead of reconstructing them later from
    element coordinates.
    """
    left = factory.addPoint(float(x[0]), float(outer_z), 0, float(lc))
    right = factory.addPoint(float(x[-1]), float(outer_z), 0, float(lc))

    if above:
        outer = factory.addLine(left, right)
        right_edge = factory.addLine(right, interface_points[-1])
        left_edge = factory.addLine(interface_points[0], left)
        edges = [
            outer,
            right_edge,
            *[-edge for edge in reversed(interface_edges)],
            left_edge,
        ]
    else:
        right_edge = factory.addLine(interface_points[-1], right)
        outer = factory.addLine(right, left)
        left_edge = factory.addLine(left, interface_points[0])
        edges = [
            *interface_edges,
            right_edge,
            outer,
            left_edge,
        ]

    surface = factory.addPlaneSurface([factory.addCurveLoop(edges)])
    return surface, {
        "outer": outer,
        "left": left_edge,
        "right": right_edge,
    }


def add_layer_variable_vs_2d(
    top_z, bottom_z, x, interp_vs, lc_values,
    top_points, bottom_points, top_edges, bottom_edges,
    n_vertical=20, vs_bin=10, min_vs=100.0, f_att=50.0,
):
    """Split sediments into cells and group them by quantized centre Vs."""
    grid = []
    for ix, xi in enumerate(x):
        column = [top_points[ix]]
        for iz in range(1, n_vertical):
            fraction = iz / n_vertical
            z = (1 - fraction) * top_z[ix] + fraction * bottom_z[ix]
            column.append(factory.addPoint(float(xi), float(z), 0, float(lc_values[ix])))
        column.append(bottom_points[ix])
        grid.append(column)

    # Create each edge once; adjacent cells reuse it with opposite orientation.
    horizontal = [top_edges]
    for iz in range(1, n_vertical):
        horizontal.append(add_edges([column[iz] for column in grid]))
    horizontal.append(bottom_edges)
    vertical = [add_edges(column) for column in grid]
    groups = {}
    all_sediment_surfaces = []
    top_interface_groups = {}
    for ix in range(len(x) - 1):
        for iz in range(n_vertical):
            loop = factory.addCurveLoop([
                horizontal[iz][ix], vertical[ix + 1][iz],
                -horizontal[iz + 1][ix], -vertical[ix][iz],
            ])
            surface = factory.addPlaneSurface([loop])
            all_sediment_surfaces.append(surface)
            xc = 0.5 * (x[ix] + x[ix + 1])
            fraction = (iz + 0.5) / n_vertical
            zc = (
                (1 - fraction) * 0.5 * (top_z[ix] + top_z[ix + 1])
                + fraction * 0.5 * (bottom_z[ix] + bottom_z[ix + 1])
            )
            vs = max(float(interp_vs(xc, zc)), min_vs)
            vs_round = max(min_vs, round(vs / vs_bin) * vs_bin)
            name = f"sed_vs_{vs_round:g}"
            groups.setdefault(name, []).append(surface)
            if iz == 0:
                # Exact water/sediment geometric curve for this material bin.
                top_interface_groups.setdefault(name, []).append(top_edges[ix])

    materials = {
        name: {
            "rho": 1800, "vp": 1800, "vs": float(name.removeprefix("sed_vs_")),
            "q_kappa": 100, "q_mu": 50, "f_att": f_att,
        }
        for name in groups
    }
    boundary_curves = {
        "left": list(vertical[0]),
        "right": list(vertical[-1]),
    }
    structured_grid = {
        "horizontal_curves": horizontal,
        "vertical_curves": vertical,
        "surfaces": all_sediment_surfaces,
    }
    return materials, groups, top_interface_groups, boundary_curves, structured_grid



def plan_transfinite_sediment(x, top_z, bottom_z, lc_values,
                             vertical_percentile=95.0):
    """Estimate a conforming structured sediment mesh before calling Gmsh.

    Every horizontal edge of the same x interval uses the same subdivisions.
    Every vertical edge of the same sediment layer uses the same subdivisions,
    across ALL columns: otherwise adjacent transfinite cells would not match.

    We use a high percentile in the vertical direction because a single very
    deep/thin outlier can force excessive subdivision of every single column.
    The under-resolution relative to the requested sizing is reported below;
    the user can select 100% to honour even the worst edge.
    """
    x = np.asarray(x, dtype=float)
    top_z = np.asarray(top_z, dtype=float)
    bottom_z = np.asarray(bottom_z, dtype=float)
    lc_values = np.asarray(lc_values, dtype=float)
    if x.ndim != 1 or not (x.shape == top_z.shape == bottom_z.shape == lc_values.shape):
        raise ValueError("Inconsistent sediment grid input shapes")
    if not (0 < vertical_percentile <= 100):
        raise ValueError("vertical_percentile must be in (0,100]")
    if np.any(lc_values <= 0):
        raise ValueError("All sediment mesh sizes must be positive")

    # Number of SEGMENTS (setTransfiniteCurve needs one more NODE than this).
    horizontal_length = np.diff(x)
    horizontal_lc = np.minimum(lc_values[:-1], lc_values[1:])
    nx = np.maximum(1, np.ceil(horizontal_length / horizontal_lc).astype(np.int32))

    # Every sediment layer spans exactly 1/N_VERTICAL of the local thickness;
    # compute a layer-independent value here, then repeat it for all layers.
    # N_VERTICAL itself is provided to the function below.
    return nx, np.asarray((top_z - bottom_z) / lc_values, dtype=float)


def configure_transfinite_sediment_mesh(structured_grid, x, top_z, bottom_z,
                                        lc_values, n_vertical,
                                        vertical_percentile=95.0,
                                        max_sediment_quads=3_500_000,
                                        dry_run=False):
    """Apply structured transfinite quads, or report costs without meshing.

    This changes only meshing constraints, not geometry/material/physical tags.
    It is conforming at cell interfaces, including water and basement.
    """
    nx, total_depth_size_ratio = plan_transfinite_sediment(
        x, top_z, bottom_z, lc_values, vertical_percentile
    )
    layer_size_ratio = total_depth_size_ratio / float(n_vertical)
    nz_per_layer = max(1, int(np.ceil(np.percentile(layer_size_ratio, vertical_percentile))))

    # The seds have exactly n_vertical rows with these nX/nZ divisions.
    estimated_quads = int(np.sum(nx, dtype=np.int64) * n_vertical * nz_per_layer)
    vertical_coarsening = layer_size_ratio / nz_per_layer
    ratio_stats = np.percentile(vertical_coarsening, [50, 90, 95, 99, 100])
    n_exceed = int(np.count_nonzero(vertical_coarsening > 1.0 + 1e-6))
    n_exceed_125 = int(np.count_nonzero(vertical_coarsening > 1.25))

    print("\n========== TRANSFINITE SEDIMENT PREFLIGHT ==========", flush=True)
    print(f"  x intervals: {len(nx):,}")
    print(f"  vertical layers: {n_vertical}")
    print(f"  horizontal subdivisions per interval: min={nx.min()}, "
          f"median={np.median(nx):g}, max={nx.max()}")
    print(f"  vertical sizing percentile: {vertical_percentile:g}%")
    print(f"  subdivisions per vertical layer: {nz_per_layer}")
    print(f"  estimated sediment quads: {estimated_quads:,}")
    print("  vertical element size / requested lc (p50/p90/p95/p99/max): "
          + " / ".join(f"{v:.2f}" for v in ratio_stats))
    print(f"  columns above requested vertical lc: {n_exceed:,}/{len(nx)+1:,}")
    print(f"  columns >1.25x requested vertical lc: {n_exceed_125:,}/{len(nx)+1:,}")
    print("====================================================\n", flush=True)

    if estimated_quads > max_sediment_quads:
        raise RuntimeError(
            f"Transfinite preflight estimates {estimated_quads:,} sediment quads "
            f"(limit {max_sediment_quads:,}). No meshing started. "
            "Review vertical_percentile and mesh resolution; do not "
            "automatically reduce subdivisions."
        )
    if dry_run:
        return estimated_quads

    horizontal = structured_grid["horizontal_curves"]
    vertical = structured_grid["vertical_curves"]
    surfaces = structured_grid["surfaces"]
    if len(horizontal) != n_vertical+1 or len(vertical) != len(nx)+1:
        raise RuntimeError("Unexpected sediment curve grid dimensions")
    if len(surfaces) != len(nx)*n_vertical:
        raise RuntimeError("Unexpected sediment surface grid dimensions")

    # The lines are shared across neighbouring surfaces and across media.
    # Set each UNIQUE curve's divisions once, avoiding conflicting constraints.
    for row in horizontal:
        if len(row) != len(nx):
            raise RuntimeError("Invalid horizontal curve row")
        for edge, n in zip(row, nx):
            gmsh.model.mesh.setTransfiniteCurve(int(edge), int(n)+1)
    for column in vertical:
        if len(column) != n_vertical:
            raise RuntimeError("Invalid vertical curve column")
        for edge in column:
            gmsh.model.mesh.setTransfiniteCurve(int(edge), nz_per_layer+1)

    # Each sediment cell has exactly four corners (one Gmsh curve per side).
    for surface in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(int(surface))
        gmsh.model.mesh.setRecombine(2, int(surface))
    print(f"Transfinite constraints applied to {len(surfaces):,} sediment surfaces.",
          flush=True)
    return estimated_quads


def configure_adaptive_water_mesh(
    water_surface,
    x,
    seafloor,
    interface_lc,
    lc_far=8.0,
    bottom_keep_distance=5.0,
    bottom_transition_distance=35.0,
    protected_points=(),
    protected_lc=5.0,
    protected_keep_radius=20.0,
    protected_transition_radius=50.0,
):
    """Coarsen only the interior of the water domain without changing geometry.

    The water/seafloor boundary keeps its original 1-D discretization. Boundary
    mesh sizes are prevented from propagating through the whole water surface,
    and a size callback controls only the interior of the water surface.

    Near the seafloor, the target size is the existing conforming interface size.
    It then grows smoothly toward ``lc_far``. Source/receiver neighbourhoods can
    independently remain refined through ``protected_points``.
    """
    x = np.asarray(x, dtype=float)
    seafloor = np.asarray(seafloor, dtype=float)
    interface_lc = np.asarray(interface_lc, dtype=float)

    if x.ndim != 1 or seafloor.shape != x.shape or interface_lc.shape != x.shape:
        raise ValueError("x, seafloor and interface_lc must be 1-D arrays with the same shape")
    if bottom_transition_distance <= bottom_keep_distance:
        raise ValueError("bottom_transition_distance must be larger than bottom_keep_distance")
    if protected_transition_radius <= protected_keep_radius:
        raise ValueError("protected_transition_radius must be larger than protected_keep_radius")
    if lc_far <= 0 or protected_lc <= 0:
        raise ValueError("Mesh sizes must be strictly positive")

    protected_points = np.asarray(protected_points, dtype=float)
    if protected_points.size == 0:
        protected_points = np.empty((0, 2), dtype=float)
    protected_points = protected_points.reshape(-1, 2)

    # Do not extend small boundary sizes through the complete water surface.
    gmsh.model.mesh.setSizeFromBoundary(2, int(water_surface), 0)

    def smoothstep01(value):
        value = max(0.0, min(1.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    def blend(size_a, size_b, distance, distance_a, distance_b):
        if distance <= distance_a:
            return float(size_a)
        if distance >= distance_b:
            return float(size_b)
        t = smoothstep01((distance - distance_a) / (distance_b - distance_a))
        return float(size_a + (size_b - size_a) * t)

    def water_size_callback(dim, tag, xq, yq, zq, lc):
        # Leave sediment, basement and all 1-D boundary discretizations untouched.
        if int(dim) != 2 or int(tag) != int(water_surface):
            return float(lc)

        bottom_z = float(np.interp(xq, x, seafloor))
        bottom_size = float(np.interp(xq, x, interface_lc))
        distance_to_bottom = max(0.0, float(yq) - bottom_z)

        target = blend(
            bottom_size,
            lc_far,
            distance_to_bottom,
            bottom_keep_distance,
            bottom_transition_distance,
        )

        # Keep source and acoustic receiver neighbourhoods refined.
        for px, py in protected_points:
            distance = float(np.hypot(float(xq) - px, float(yq) - py))
            local_target = blend(
                protected_lc,
                lc_far,
                distance,
                protected_keep_radius,
                protected_transition_radius,
            )
            target = min(target, local_target)

        return float(target)

    gmsh.model.mesh.setSizeCallback(water_size_callback)

    print("Adaptive water mesh:")
    print(f"  far-water target size: {lc_far:g} m")
    print(
        "  bottom transition: "
        f"{bottom_keep_distance:g} m -> {bottom_transition_distance:g} m"
    )
    print(
        "  protected-point transition: "
        f"{protected_keep_radius:g} m -> {protected_transition_radius:g} m"
    )
    print(f"  protected-point target size: {protected_lc:g} m")
    print(f"  protected points: {len(protected_points)}")

def add_physical_group(surfaces, tag, name):
    gmsh.model.addPhysicalGroup(2, surfaces, tag)
    gmsh.model.setPhysicalName(2, tag, name)


def add_physical_curve_group(curves, tag, name):
    """Create an explicit 1-D physical group used as an Exodus side set."""
    curves = list(dict.fromkeys(int(c) for c in curves))
    if not curves:
        raise ValueError(f"Physical curve group {name!r} is empty")
    gmsh.model.addPhysicalGroup(1, curves, tag)
    gmsh.model.setPhysicalName(1, tag, name)


def lc_from_vs(vs, fmax=50.0, nppw=6, lc_min=1.0, lc_max=20.0):
    return np.clip(np.asarray(vs) / (fmax * nppw), lc_min, lc_max)



def _count_2d_element_types_on_surfaces(surface_tags):
    """Return triangle/quad counts for the supplied Gmsh 2-D surface tags."""
    counts = {"triangles": 0, "quads": 0, "other": 0}
    per_surface = []

    for surface_tag in surface_tags:
        types, element_tags, _ = gmsh.model.mesh.getElements(2, int(surface_tag))
        ntri = 0
        nquad = 0
        nother = 0

        for element_type, tags in zip(types, element_tags):
            n = len(tags)
            if int(element_type) == 2:
                ntri += n
            elif int(element_type) == 3:
                nquad += n
            else:
                nother += n

        counts["triangles"] += ntri
        counts["quads"] += nquad
        counts["other"] += nother
        per_surface.append((int(surface_tag), ntri, nquad, nother))

    return counts, per_surface


def report_mesh_element_distribution(water_surface, sediment_surfaces, rock_surface):
    """Print where the remaining triangles are located after 2-D meshing."""
    categories = {
        "water": [int(water_surface)],
        "sediment": [int(tag) for tag in sediment_surfaces],
        "basement": [int(rock_surface)],
    }

    print()
    print("========================================")
    print(" 2-D ELEMENT DISTRIBUTION DIAGNOSTIC")
    print("========================================")

    total_tri = 0
    total_quad = 0
    total_other = 0
    sediment_detail = []

    for name, surfaces in categories.items():
        counts, detail = _count_2d_element_types_on_surfaces(surfaces)
        total_tri += counts["triangles"]
        total_quad += counts["quads"]
        total_other += counts["other"]

        print(
            f"{name:10s}: "
            f"{counts['triangles']:9d} triangles | "
            f"{counts['quads']:9d} quads | "
            f"{counts['other']:6d} other | "
            f"{len(surfaces):6d} surfaces"
        )

        if name == "sediment":
            sediment_detail = detail

    total = total_tri + total_quad + total_other
    tri_pct = (100.0 * total_tri / total) if total else 0.0

    print("----------------------------------------")
    print(
        f"{'TOTAL':10s}: "
        f"{total_tri:9d} triangles | "
        f"{total_quad:9d} quads | "
        f"{total_other:6d} other"
    )
    print(f"Triangle fraction among 2-D elements: {tri_pct:.4f}%")

    if sediment_detail:
        ranked = sorted(
            sediment_detail,
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )
        with_triangles = [row for row in ranked if row[1] > 0]

        print()
        print(
            "Sediment surfaces containing triangles: "
            f"{len(with_triangles)} / {len(sediment_detail)}"
        )
        if with_triangles:
            print("Top 20 sediment surfaces by triangle count:")
            print("  surface_tag   triangles      quads      other")
            for tag, ntri, nquad, nother in with_triangles[:20]:
                print(
                    f"  {tag:11d}   {ntri:9d}   "
                    f"{nquad:9d}   {nother:8d}"
                )

    print("========================================")
    print()

def _element_geometry(element_id):
    """Independent corner geometry for a suspicious Gmsh quad."""
    typ, node_ids, dim, entity = gmsh.model.mesh.getElement(int(element_id))
    if int(typ) != 3 or len(node_ids) != 4:
        raise RuntimeError(f"Unexpected element type/connectivity for {element_id}")
    pts = np.asarray([
        gmsh.model.mesh.getNode(int(nid))[0][:2] for nid in node_ids
    ], dtype=float)
    delta = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(delta, axis=1)
    area = 0.5 * float(np.sum(
        pts[:, 0]*np.roll(pts[:, 1], -1)
        - pts[:, 1]*np.roll(pts[:, 0], -1)))
    # Corner signed Jacobians of the bilinear map (up to a common factor 1/4).
    nxt = np.roll(pts, -1, axis=0) - pts
    prv = pts - np.roll(pts, 1, axis=0)
    corner_j = nxt[:, 0]*prv[:, 1] - nxt[:, 1]*prv[:, 0]
    center = pts.mean(axis=0)
    return {
        "surface": int(entity), "dim": int(dim),
        "x_center": float(center[0]), "z_center": float(center[1]),
        "signed_area": area, "min_edge_length": float(lengths.min()),
        "min_corner_cross": float(corner_j.min()),
        "max_corner_cross": float(corner_j.max()),
        "nodes": " ".join(str(int(n)) for n in node_ids),
    }


def inspect_quad_quality(quality_threshold=0.05, warning_threshold=0.10,
                         chunk_size=200_000, report_csv=None, label=""):
    """Check ALL quads using minSICN; save all low-quality elements for diagnosis.

    The thresholds are deliberately conservative for explicit wave propagation:
    quads with 0 < Q < 0.05 still need repair, not just negative elements.
    """
    element_types, element_tags, _ = gmsh.model.mesh.getElements(2)
    quad_arrays = [tags for typ, tags in zip(element_types, element_tags)
                   if int(typ) == 3]
    if len(quad_arrays) != 1 or len(element_types) != 1:
        raise RuntimeError("Mesh is not exclusively first-order quadrilaterals")
    ids = quad_arrays[0]
    n_failed = n_warning = 0
    min_quality = float("inf")
    low = []
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start:start + chunk_size]
        q = np.asarray(gmsh.model.mesh.getElementQualities(
            chunk_ids, "minSICN"), dtype=float)
        flagged = (~np.isfinite(q)) | (q < warning_threshold)
        failed = (~np.isfinite(q)) | (q <= quality_threshold)
        n_failed += int(np.count_nonzero(failed))
        n_warning += int(np.count_nonzero(flagged))
        if len(q):
            min_quality = min(min_quality, float(np.nanmin(q)))
        low.extend((float(v), int(e)) for v, e in zip(q[flagged], chunk_ids[flagged]))
    low.sort(key=lambda t: (-float("inf") if not np.isfinite(t[0]) else t[0]))

    per_surface = {}
    detailed = []
    for q, element_id in low:
        data = _element_geometry(element_id)
        data.update({"element": element_id, "minSICN": q})
        detailed.append(data)
        per_surface[data["surface"]] = per_surface.get(data["surface"], 0) + 1

    print(f"\n========== QUAD QUALITY: {label} ==========", flush=True)
    print(f"  quads: {len(ids):,}")
    print(f"  minSICN minimum: {min_quality:.6e}")
    print(f"  below acceptance (<= {quality_threshold}): {n_failed:,}")
    print(f"  below warning (< {warning_threshold}): {n_warning:,}")
    print(f"  affected surfaces: {sorted(per_surface.items())}")
    print("  worst elements: Q, id, surface, (x,z), signed area, min edge, min corner cross")
    for d in detailed[:15]:
        print(f"    {d['minSICN']:.3e}  {d['element']}  {d['surface']}  "
              f"({d['x_center']:.3f},{d['z_center']:.3f})  "
              f"A={d['signed_area']:.6e}  edge={d['min_edge_length']:.6e}  "
              f"Jmin={d['min_corner_cross']:.6e}")
    print("====================================================\n", flush=True)

    if report_csv is not None:
        report_csv = Path(report_csv)
        fieldnames = ["element", "surface", "dim", "minSICN", "x_center", "z_center",
                      "signed_area", "min_edge_length", "min_corner_cross",
                      "max_corner_cross", "nodes"]
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed)
        print(f"Detailed quality CSV: {report_csv}", flush=True)
    return {"n_failed": n_failed, "n_warning": n_warning,
            "min_quality": min_quality, "n_quads": len(ids)}


def validate_quad_quality(quality_threshold=0.05, warning_threshold=0.10,
                          chunk_size=200_000):
    """Final gate: do not export very poor quads to explicit wave solver."""
    stats = inspect_quad_quality(quality_threshold, warning_threshold,
                                 chunk_size, label="final acceptance")
    if stats["n_failed"]:
        raise RuntimeError(f"Quad quality validation failed: {stats['n_failed']} bad quads")


def _assert_interface_curve_sharing(water_surface, rock_surface, top_curves, bottom_curves):
    """Verify that both media reuse *identical* geometric interface curves.

    The 1-D discretization belongs to these curves, not separately to each
    surface; Gmsh then shares their nodes across 2-D domains.
    """
    top_curves = [int(v) for v in top_curves]
    bottom_curves = [int(v) for v in bottom_curves]
    if not top_curves or not bottom_curves:
        raise RuntimeError("Missing interface curve tags")
    for domain, boundary_curves, label in (
        (int(water_surface), top_curves, "water/sediment"),
        (int(rock_surface), bottom_curves, "sediment/basement"),
    ):
        outer_boundary = {
            int(t) for dim, t in gmsh.model.getBoundary(
                [(2, domain)], combined=False, oriented=False
            ) if int(dim) == 1
        }
        missing = set(boundary_curves) - outer_boundary
        if missing:
            raise RuntimeError(
                f"{label}: {len(missing)} interface curves not in the outer "
                f"surface boundary, e.g. {sorted(missing)[:8]}"
            )
        # A sediment cell is adjacent to each of the same interface curves.
        # Check the geometric adjacency for *all* curve tags (not just endpoints).
        bad = []
        for tag in boundary_curves:
            adjacent, _ = gmsh.model.getAdjacencies(1, tag)
            adjacent = {int(t) for t in adjacent}
            if domain not in adjacent or len(adjacent) != 2:
                bad.append((tag, tuple(sorted(adjacent))))
        if bad:
            raise RuntimeError(
                f"{label}: {len(bad)} curves have incorrect surface adjacency; "
                f"examples: {bad[:5]}"
            )
        print(f"INTERFACE GEOMETRY OK {label}: {len(boundary_curves):,} shared curves", flush=True)


def _assert_interface_quad_edges(water_surface, rock_surface,
                                 top_curves, bottom_curves):
    """Every interface 1-D segment must be an edge of adjacent outer elements.

    The sediment side uses the exact same Gmsh curves and transfinite nodes.
    We check both triangles and quads so even rejected candidates are audited.
    """
    def encode(a, b):
        aa = np.minimum(a, b).astype(np.uint64, copy=False)
        bb = np.maximum(a, b).astype(np.uint64, copy=False)
        if (len(aa) and np.max(bb) >= 2**32):
            raise RuntimeError("Node tags >= 2^32 are not supported by interface audit")
        return (aa << np.uint64(32)) | bb

    def keys_from_interface_curves(curves):
        parts = []
        for curve in curves:
            types, tags, nodes = gmsh.model.mesh.getElements(1, int(curve))
            if not types or any(int(t) != 1 for t in types):
                raise RuntimeError(f"No first-order line elements on interface curve {curve}")
            for conn in nodes:
                xy = np.asarray(conn, dtype=np.uint64).reshape(-1, 2)
                parts.append(encode(xy[:, 0], xy[:, 1]))
        return np.concatenate(parts)

    def keys_from_surface_elements(surface):
        types, tags, nodes = gmsh.model.mesh.getElements(2, int(surface))
        pieces = []
        for element_type, flat in zip(types, nodes):
            if int(element_type) not in (2, 3):
                raise RuntimeError(f"Surface {surface}: unexpected element type {element_type}")
            corners = 3 if int(element_type) == 2 else 4
            conn = np.asarray(flat, dtype=np.uint64).reshape(-1, corners)
            for i in range(corners):
                pieces.append(encode(conn[:, i], conn[:, (i + 1) % corners]))
        if not pieces:
            raise RuntimeError(f"Surface {surface}: no 2-D elements")
        return np.unique(np.concatenate(pieces))

    for label, surface, curves in (
        ("water/sediment", water_surface, top_curves),
        ("sediment/basement", rock_surface, bottom_curves),
    ):
        boundary_edges = keys_from_interface_curves(curves)
        if len(np.unique(boundary_edges)) != len(boundary_edges):
            raise RuntimeError(f"{label}: duplicate 1-D boundary segment")
        surface_edges = keys_from_surface_elements(surface)
        positions = np.searchsorted(surface_edges, boundary_edges)
        valid = positions < len(surface_edges)
        valid[valid] = surface_edges[positions[valid]] == boundary_edges[valid]
        missing = int(np.count_nonzero(~valid))
        if missing:
            raise RuntimeError(
                f"{label}: {missing} / {len(boundary_edges)} curve edges are "
                "missing from the adjacent quadrilateral surface"
            )
        print(f"INTERFACE MESH OK {label}: {len(boundary_edges):,} matching 1-D/2-D edges", flush=True)




# ---------------------------------------------------------------------------
# Local all-quad repair: first existing-node quad patches, then Steiner patches
# Based on the successful hybrid5 -> retile -> Steiner workflow.
# No global optimization; no modifications to original 1D nodes/edges.
# ---------------------------------------------------------------------------

def E(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def edges(row):
    return [E(row[j], row[(j + 1) % len(row)]) for j in range(len(row))]


def K(a, b):
    return (np.minimum(a, b).astype(np.uint64) << np.uint64(32)) | np.maximum(a, b).astype(np.uint64)


def ring_from_cells(cells):
    counts = Counter(e for row in cells for e in edges(row))
    if any(c > 2 for c in counts.values()):
        return None
    boundary = [e for e, c in counts.items() if c == 1]
    graph = defaultdict(list)
    for a, b in boundary:
        graph[a].append(b)
        graph[b].append(a)
    if len(graph) != len(boundary) or any(len(nei) != 2 for nei in graph.values()):
        return None
    start = min(graph)
    ring = [start, min(graph[start])]
    while len(ring) < len(graph):
        options = [v for v in graph[ring[-1]] if v != ring[-2]]
        if len(options) != 1 or options[0] in ring:
            return None
        ring.append(options[0])
    if start not in graph[ring[-1]]:
        return None
    return ring


def coordinates(node, cache):
    node = int(node)
    if node not in cache:
        p, _, _, _ = gmsh.model.mesh.getNode(node)
        cache[node] = np.asarray(p[:2], dtype=np.float64)
    return cache[node]


def area(row, coord):
    p = np.asarray([coordinates(v, coord) for v in row])
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) * .5)


def corner_quality(row, coord, orient):
    if len(set(row)) != 4:
        return -1.
    p = np.asarray([coordinates(v, coord) for v in row])
    prev = p - np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0) - p
    norms = np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
    if np.min(norms) < 1e-14:
        return -1.
    cross = prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]
    return float(np.min(orient * cross / norms))


def quadtilings(ring):
    """All ear-cut quadrangulations of even simple boundary polygon (4,6,8)."""
    n = len(ring)
    if n == 4:
        yield (tuple(ring),)
        return
    if n < 4 or n % 2:
        return
    for i in range(n):
        ear = tuple(ring[(i + j) % n] for j in range(4))
        rest = [ring[i]] + [ring[(i + j) % n] for j in range(3, n)]
        for tail in quadtilings(rest):
            yield (ear,) + tail


def segment_intersect(a, b, c, d, tol=1e-12):
    """True for proper intersection; collinear overlaps handled separately."""
    def cross(p, q, r):
        return float((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))
    v = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    return v[0] * v[1] < -tol and v[2] * v[3] < -tol


def point_inside_polygon(p, ring, coord):
    """Ray-casting, counting points on edges as inside."""
    pts = [coordinates(i, coord) for i in ring]
    px, py = float(p[0]), float(p[1])
    inside = False
    for i, a in enumerate(pts):
        b = pts[(i+1) % len(pts)]
        v = b-a
        cross = v[0]*(py-a[1])-v[1]*(px-a[0])
        dot = (px-a[0])*(px-b[0])+(py-a[1])*(py-b[1])
        if abs(cross) <= 1e-9 and dot <= 1e-9:
            return True
        if (a[1]>py) != (b[1]>py):
            xx = a[0] + (py-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if xx > px:
                inside = not inside
    return inside


def tiling_inside_polygon(tiling, ring, coord):
    rim_edges = set(edges(ring))
    for quad in tiling:
        for e in edges(quad):
            if e in rim_edges:
                continue
            a, b = coordinates(e[0], coord), coordinates(e[1], coord)
            # Check candidate internal diagonals, which must stay inside domain.
            for frac in (0.2, 0.5, 0.8):
                if not point_inside_polygon(a*(1-frac)+b*frac, ring, coord):
                    return False
    return True


def boundary_edges(cells):
    return {e for e,c in Counter(e for row in cells for e in edges(row)).items() if c==1}


def valid_simple_ring(ring, coord):
    pts = [coordinates(i, coord) for i in ring]
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        if np.linalg.norm(a-b) < 1e-10:
            return False
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            if segment_intersect(a, b, pts[j], pts[(j+1) % n]):
                return False
    return True


def best_tiling(cells, coord, orientation, threshold):
    ring = ring_from_cells(cells)
    if ring is None or len(ring) not in (4, 6, 8) or not valid_simple_ring(ring, coord):
        return None
    orig_area = sum(abs(area(cell, coord)) for cell in cells)
    outer_area = area(ring, coord)
    if outer_area * orientation < 0:
        ring.reverse()
        outer_area = -outer_area
    if outer_area * orientation <= 1e-10:
        return None
    if not math.isclose(abs(outer_area), orig_area, rel_tol=5e-7, abs_tol=1e-5):
        return None
    seen = set()
    best = None
    for tiling in quadtilings(ring):
        key = tuple(sorted(tuple(sorted(row)) for row in tiling))
        if key in seen:
            continue
        seen.add(key)
        qualities = [corner_quality(row, coord, orientation) for row in tiling]
        worst = min(qualities)
        if worst <= threshold:
            continue
        if not math.isclose(sum(area(q, coord) for q in tiling), outer_area,
                            rel_tol=5e-7, abs_tol=1e-5):
            continue
        if boundary_edges(tiling) != boundary_edges(cells):
            continue
        if not tiling_inside_polygon(tiling, ring, coord):
            continue
        if best is None or worst > best[0]:
            best = (worst, tuple(tuple(int(i) for i in q) for q in tiling))
    return best


def surface_cells(surface):
    types, ids, conn = gmsh.model.mesh.getElements(2, surface)
    values = {}
    for t, tags, nodes in zip(types, ids, conn):
        t = int(t)
        if t not in (2, 3):
            raise RuntimeError(f'Unexpected 2D element type {t} on {surface}')
        values[t] = (np.asarray(tags, dtype=np.int64),
                     np.asarray(nodes, dtype=np.int64).reshape(-1, t + 1))
    return values


def outer_surfaces():
    """Find the two outer entities from physical names (not fragile tag numbers)."""
    found={}
    for dim, phys in gmsh.model.getPhysicalGroups(2):
        name=gmsh.model.getPhysicalName(dim, phys)
        label={'socle_1': 'water', 'socle_3': 'basement'}.get(name)
        if label is not None:
            entities=gmsh.model.getEntitiesForPhysicalGroup(dim, phys)
            if len(entities) != 1:
                raise RuntimeError(f'{name} must contain exactly one 2D surface')
            found[label]=int(entities[0])
    if len(found)!=2:
        raise RuntimeError(f'Cannot resolve physical water and basement: {found}')
    return found


def eligible_neighbors(tri, quad_ids, quad_conn):
    """Return mapping boundary edge -> (quad element ID, nodes) without large Python adjacency maps."""
    triangle_edge_keys = set()
    for row in tri:
        for a, b in edges(row):
            triangle_edge_keys.add((a << 32) | b)
    keys = np.asarray(sorted(triangle_edge_keys), dtype=np.uint64)
    result = defaultdict(dict)
    for k in range(4):
        v = K(quad_conn[:, k], quad_conn[:, (k+1) % 4])
        selected = np.nonzero(np.isin(v, keys, assume_unique=False, kind='sort'))[0]
        for i in selected:
            row = tuple(int(n) for n in quad_conn[i])
            result[int(v[i])][int(quad_ids[i])] = row
    return result


def scan_surface(name, surf, threshold, max_extra):
    cells = surface_cells(surf)
    ti, tri = cells.get(2, (np.empty(0, np.int64), np.empty((0,3), np.int64)))
    qi, quad = cells.get(3, (np.empty(0, np.int64), np.empty((0,4), np.int64)))
    print(f'\n{name}: {len(ti):,} triangles and {len(qi):,} quads', flush=True)
    if not len(ti):
        return [], 0
    sample = np.linspace(0, len(qi)-1, min(30,len(qi)), dtype=int)
    nodecache = {}
    ori = 1 if sum(1 if area(quad[i], nodecache)>0 else -1 for i in sample) >= 0 else -1
    tri_owner = defaultdict(list)
    tri_dict = {int(t): tuple(int(v) for v in n) for t,n in zip(ti,tri)}
    for tid, nodes in tri_dict.items():
        for e in edges(nodes):
            tri_owner[e].append(tid)
    pairs = [(tuple(tags), e) for e, tags in tri_owner.items() if len(tags) == 2]
    quad_neis = eligible_neighbors(tri, qi, quad)
    candidates = []
    best_per_pair = []
    for pair, _ in pairs:
        tri_rows = [tri_dict[t] for t in pair]
        rim = [e for e,c in Counter(edges(tri_rows[0])+edges(tri_rows[1])).items() if c==1]
        nei = {}
        for a,b in rim:
            nei.update(quad_neis.get((a<<32)|b, {}))
        best = None
        for extra_count in range(0, min(max_extra, len(nei))+1):
            for extra_ids in combinations(nei, extra_count):
                rows = tri_rows + [nei[eid] for eid in extra_ids]
                tiling = best_tiling(rows, nodecache, ori, threshold)
                if tiling is None:
                    continue
                score, new_quads = tiling
                item = (float(score), frozenset(pair + extra_ids), tuple(extra_ids), tuple(pair), new_quads)
                if best is None or item[0] > best[0]:
                    best = item
        if best is not None:
            best_per_pair.append(best)
            candidates.append(best)
    # Disjoint greedy patches, most promising first
    selected, reserved = [], set()
    for candidate in sorted(candidates, key=lambda x:x[0], reverse=True):
        _, affected, _, _, _ = candidate
        if not affected.intersection(reserved):
            selected.append(candidate)
            reserved.update(affected)
    covered_triangles = len(selected)*2
    print(f'  paired triangles: {len(pairs)}, candidates for {len(best_per_pair)} pairs', flush=True)
    print(f'  non-overlapping patches: {len(selected)}, covering {covered_triangles}/{len(ti)} triangles', flush=True)
    for sc, _, add, pair, quads in sorted(selected, key=lambda x:x[0], reverse=True)[:8]:
        print(f'    predicted quality={sc:.3f}: triangles={pair}, adjacent quads={add}, replacements={len(quads)}')
    return selected, len(ti)


def quality_star(boundary, center, ori):
    """Return corner-sine proxy and quad vertex arrays (point-ids; -1=center)."""
    n = len(boundary)
    quads = []
    mins = []
    for k in range(0, n, 2):
        # Fan uses center and every second BOUNDARY vertex as spokes.
        p = np.stack((center, boundary[k], boundary[(k + 1) % n],
                      boundary[(k + 2) % n]), axis=0)
        prv = p - np.roll(p, 1, axis=0)
        nxt = np.roll(p, -1, axis=0) - p
        denom = np.linalg.norm(prv, axis=1) * np.linalg.norm(nxt, axis=1)
        if np.min(denom) < 1e-12:
            return -1.0
        cross = prv[:, 0] * nxt[:, 1] - prv[:, 1] * nxt[:, 0]
        vals = ori * cross / denom
        mins.append(float(np.min(vals)))
    return min(mins) if mins else -1.0


def polygon_centroid(points):
    p = np.asarray(points)
    cross = p[:, 0]*np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1)*p[:, 1]
    a = cross.sum()
    if abs(a) < 1e-10:
        return p.mean(axis=0)
    x = ((p[:, 0]+np.roll(p[:, 0],-1))*cross).sum() / (3*a)
    y = ((p[:, 1]+np.roll(p[:, 1],-1))*cross).sum() / (3*a)
    return np.array([x, y], float)


def best_star(cells, cache, ori, threshold, grid_n=11):
    ring = ring_from_cells(cells)
    if ring is None or len(ring) < 6 or len(ring) % 2 or len(ring) > 12:
        return None
    if not valid_simple_ring(ring, cache):
        return None
    signed = area(ring, cache)
    orig = sum(abs(area(cell, cache)) for cell in cells)
    if signed * ori < 0:
        ring.reverse()
        signed = -signed
    if signed * ori <= 1e-10:
        return None
    if not math.isclose(abs(signed), orig, abs_tol=1e-5, rel_tol=5e-7):
        return None
    xy = np.asarray([coordinates(v, cache) for v in ring], dtype=float)
    ctr = polygon_centroid(xy)
    mean = xy.mean(axis=0)
    extent = np.ptp(xy, axis=0)
    if np.min(extent) <= 1e-12:
        return None
    # Search in principal coordinates (important for very thin, tilted patches).
    rot = np.linalg.eigh(np.cov(xy.T))[1]
    trans = (xy - mean) @ rot
    lo = trans.min(axis=0)
    hi = trans.max(axis=0)
    uu = np.linspace(.06, .94, grid_n)
    points = [ctr, mean]
    points += [xy.mean(axis=0) * (1-a) + v*a for v in xy for a in (.2, .4, .6)]
    for u in uu:
        for v in uu:
            points.append(mean + np.array([lo[0]*(1-u) + hi[0]*u,
                                           lo[1]*(1-v) + hi[1]*v]) @ rot.T)
    # Try both checkerboard parities; parity 1 rotates polygon by one vertex.
    best = None
    for parity in (0, 1):
        rr = list(ring[parity:] + ring[:parity])
        pp = np.asarray([coordinates(v, cache) for v in rr], dtype=float)
        trial_best = (-1.0, None)
        for center in points:
            if not point_inside_polygon(center, rr, cache):
                continue
            q = quality_star(pp, center, ori)
            if q > trial_best[0]:
                trial_best = (q, np.asarray(center, dtype=float))
        if trial_best[1] is None:
            continue
        # Refine best interior-point position without affecting any existing node.
        center = trial_best[1].copy()
        score = trial_best[0]
        step = np.maximum(hi-lo, 1e-8) / max(4, grid_n//2)
        for level in range(7):
            improved = True
            while improved:
                improved = False
                for du, dv in ((1,0),(-1,0),(0,1),(0,-1),(1,1),
                               (1,-1),(-1,1),(-1,-1)):
                    proposed = center + np.array([du*step[0], dv*step[1]]) @ rot.T
                    if not point_inside_polygon(proposed, rr, cache):
                        continue
                    q = quality_star(pp, proposed, ori)
                    if q > score + 1e-6:
                        center, score, improved = proposed, q, True
            step *= .5
        if score <= threshold:
            continue
        # New quads share one NEW surface-interior vertex; ring edges preserved.
        with_center = [(-1, int(rr[k]), int(rr[(k+1) % len(rr)]),
                        int(rr[(k+2) % len(rr)])) for k in range(0,len(rr),2)]
        # Area and boundary checks prevent a topological mismatch / gap.
        tmp = dict(cache)
        tmp[-1] = center
        test_rows = [(-1, *q[1:]) for q in with_center]
        if boundary_edges(test_rows) != boundary_edges(cells):
            continue
        if not math.isclose(sum(area(row, tmp) for row in test_rows),
                            signed, rel_tol=5e-7, abs_tol=1e-5):
            continue
        # No spoke may exit the old patch; no boundary edges may be modified.
        if not tiling_inside_polygon(test_rows, rr, tmp):
            continue
        trial = (float(score), tuple(int(v) for v in rr), center,
                 tuple(with_center))
        if best is None or score > best[0]:
            best = trial
    return best


def best_collared_star(cells, cache, ori, threshold, grid_n=7):
    """Robust fallback: preserve the patch boundary and insert only interior nodes.

    An even-sided patch is converted to a ring of n quadrangles plus n/2
    quadrangles meeting at a new center, using n new inner-ring nodes.
    Alternating the radial offsets removes the zero-angle problem caused by
    collinear consecutive boundary vertices (common in Gmsh triangulation).
    """
    ring = ring_from_cells(cells)
    if ring is None or len(ring) < 6 or len(ring) % 2 or len(ring) > 12:
        return None
    if not valid_simple_ring(ring, cache):
        return None
    signed = area(ring,cache)
    orig = sum(abs(area(c,cache)) for c in cells)
    if signed*ori < 0:
        ring.reverse()
        signed = -signed
    if signed*ori <= 1e-10:
        return None
    if not math.isclose(abs(signed),orig,abs_tol=1e-5,rel_tol=5e-7):
        return None
    pts = np.asarray([coordinates(v,cache) for v in ring],float)
    n=len(ring)
    mean = pts.mean(axis=0)
    centers=[polygon_centroid(pts),mean]
    # Rotated search grid: geometric candidates near thin inclined strips.
    rot=np.linalg.eigh(np.cov(pts.T))[1]
    tr=(pts-mean)@rot
    lo=tr.min(axis=0); hi=tr.max(axis=0)
    for a in np.linspace(.12,.88,grid_n):
        for d in np.linspace(.12,.88,grid_n):
            centers.append(mean + np.array([lo[0]*(1-a)+hi[0]*a,
                                            lo[1]*(1-d)+hi[1]*d]) @ rot.T)
    best=None
    for offset in (0,1):
        rr=ring[offset:]+ring[:offset]
        boundary=pts[np.r_[offset:n,0:offset]]
        # Quad templates on n boundary vertices and n inner vertices.
        # Placeholder -1 is the center; -2..-(n+1) the inner ring.
        outer=[(int(rr[i]),int(rr[(i+1)%n]),-2-((i+1)%n),-2-i)
               for i in range(n)]
        center_quads=[(-1,-2-i,-2-((i+1)%n),-2-((i+2)%n))
                      for i in range(0,n,2)]
        allquads=tuple(outer+center_quads)
        for c in centers:
            if not point_inside_polygon(c, rr, cache):
                continue
            for base in (.28,.43,.58,.70):
                for delta in (0.,.08,.16,.24,-.08,-.16,-.24):
                    t=np.array([np.clip(base+(delta if i%2==0 else -delta),
                                        .09,.85) for i in range(n)])
                    inner=boundary + (c-boundary)*t[:,None]
                    coords=np.concatenate((c.reshape(1,2),inner),axis=0)
                    # Compute all corner Jacobians from all replacement quads.
                    tmp_pts=np.concatenate((boundary,coords),axis=0)
                    # Faster explicit index mapping (positive tags are boundary).
                    mapping={int(tag):k for k,tag in enumerate(rr)}
                    mapping.update({-1:n})
                    mapping.update({-2-k:n+1+k for k in range(n)})
                    qq=np.asarray([[mapping[int(j)] for j in q] for q in allquads],int)
                    p=tmp_pts[qq]
                    prv=p-np.roll(p,1,axis=1)
                    nxt=np.roll(p,-1,axis=1)-p
                    denom=np.linalg.norm(prv,axis=2)*np.linalg.norm(nxt,axis=2)
                    if np.min(denom)<1e-12:
                        continue
                    cross=prv[:,:,0]*nxt[:,:,1]-prv[:,:,1]*nxt[:,:,0]
                    score=float(np.min(ori*cross/denom))
                    if score <= threshold or (best is not None and score <= best[0]):
                        continue
                    # Full topology and geometry checks for finalists only.
                    tmp=dict(cache)
                    tmp[-1]=np.asarray(c,float)
                    for k,v in enumerate(inner): tmp[-2-k]=v
                    if boundary_edges(allquads)!=boundary_edges(cells):
                        continue
                    if not tiling_inside_polygon(allquads,rr,tmp):
                        continue
                    if not math.isclose(sum(area(q,tmp) for q in allquads),
                                        signed,abs_tol=1e-5,rel_tol=5e-7):
                        continue
                    best=(score,tuple(int(x) for x in rr),
                          tuple(np.asarray(v,float) for v in coords), allquads)
    return best


def _choose_disjoint_patch_variants(groups, max_search=250_000):
    """Choose one non-overlapping variant per triangle pair, if possible.

    Maximize COVERAGE first. A quality-only greedy choice can block a nearby
    repair even when another valid patch is available for the first pair.
    Each group contains alternative patches for exactly one triangle pair.
    """
    if not groups:
        return []
    # Prefer patches consuming fewer neighbors; use quality to break ties.
    options = [sorted(group, key=lambda c: (len(c[1]), -c[0]))
               for group in groups]
    used = set()
    chosen = {}
    remaining = tuple(range(len(groups)))
    states = 0
    exhausted = False

    def solve(todo):
        nonlocal states, exhausted
        states += 1
        if states > max_search:
            exhausted = True
            return False
        if not todo:
            return True
        # Most-constrained pair first, recomputed as neighbors are reserved.
        best_i = None
        best_free = None
        for i in todo:
            available = [c for c in options[i] if used.isdisjoint(c[1])]
            if not available:
                return False
            if best_free is None or len(available) < len(best_free):
                best_i, best_free = i, available
                if len(best_free) == 1:
                    break
        following = tuple(i for i in todo if i != best_i)
        for cand in best_free:
            used.update(cand[1])
            chosen[best_i] = cand
            if solve(following):
                return True
            chosen.pop(best_i, None)
            used.difference_update(cand[1])
            if exhausted:
                return False
        return False

    if solve(remaining):
        return [chosen[i] for i in sorted(chosen)]

    # Safe fallback if the full non-overlapping cover does not exist or the
    # search budget is reached. Never force overlapping modifications.
    ordered = sorted(remaining, key=lambda i: (len(options[i]),
                                               min(len(c[1]) for c in options[i])))
    best = []
    for order in (ordered, list(reversed(ordered))):
        occupied = set()
        trial = []
        for i in order:
            for cand in options[i]:
                if occupied.isdisjoint(cand[1]):
                    occupied.update(cand[1])
                    trial.append(cand)
                    break
        if len(trial) > len(best):
            best = trial
    print(f'  patch-selector: full cover unavailable or search-limited '
          f'(explored {states:,} states); safe coverage={len(best)}/{len(groups)}',
          flush=True)
    return best


def pick_patches(name, surf, threshold, max_extra, grid_n):
    cells = surface_cells(surf)
    tids, tris = cells.get(2, (np.empty(0,np.int64),np.empty((0,3),np.int64)))
    qids, quads = cells.get(3, (np.empty(0,np.int64),np.empty((0,4),np.int64)))
    print(f'\n{name}: {len(tids):,} triangles, {len(qids):,} quads', flush=True)
    if len(tids) == 0:
        return [], 0
    caches = {}
    sample = np.linspace(0,len(qids)-1,min(len(qids),40),dtype=int)
    ori = 1 if sum(1 if area(quads[i],caches)>0 else -1 for i in sample)>=0 else -1
    tri_rows = {int(eid):tuple(map(int,row)) for eid,row in zip(tids,tris)}
    tri_owner = defaultdict(list)
    for eid,row in tri_rows.items():
        for e in edges(row):
            tri_owner[e].append(eid)
    pairs = [(tuple(owner),e) for e,owner in tri_owner.items() if len(owner)==2]
    quad_neis = eligible_neighbors(tris,qids,quads)
    groups=[]
    candidate_pairs=0
    for i,(pair,_) in enumerate(pairs,1):
        rows = [tri_rows[t] for t in pair]
        rim = boundary_edges(rows)
        neis={}
        for x,y in rim:
            neis.update(quad_neis.get((int(x)<<32)|int(y),{}))
        variants = {}
        best_score = -1.
        # Keep ALTERNATIVE valid neighbor sets for the same triangle pair.
        # The original code kept only the highest-scoring one, causing the
        # last pair to be blocked by a competing patch (106/107 selected).
        for k in range(1, min(max_extra,len(neis))+1):
            for ids in combinations(neis,k):
                patch_rows = rows + [neis[j] for j in ids]
                affected = frozenset((*pair,*ids))
                cand = best_star(patch_rows, caches, ori, threshold, grid_n)
                if cand is not None:
                    score, ring, ctr, outquads = cand
                    item = (float(score), affected, pair, ids,
                            (np.asarray(ctr),), outquads)
                    old = variants.get(affected)
                    if old is None or item[0] > old[0]:
                        variants[affected] = item
                    best_score = max(best_score, score)
                # The expensive collared fallback is only needed when a
                # good simple-star candidate has not yet been found.
                if best_score >= .25:
                    continue
                collared = best_collared_star(patch_rows,caches,ori,threshold,
                                             min(grid_n,7))
                if collared is not None:
                    score,ring,allnew,outquads=collared
                    item=(float(score),affected,pair,ids,allnew,outquads)
                    old=variants.get(affected)
                    if old is None or item[0]>old[0]:
                        variants[affected]=item
                    best_score = max(best_score, score)
        if variants:
            groups.append(list(variants.values()))
            candidate_pairs += 1
        if i%25==0 or i==len(pairs):
            print(f'  processed {i}/{len(pairs)} pairs, viable={candidate_pairs}',flush=True)
    chosen = _choose_disjoint_patch_variants(groups)
    print(f'  viable pairs={candidate_pairs}/{len(pairs)}; '
          f'disjoint patches={len(chosen)}, triangles covered={2*len(chosen)}/{len(tids)}',
          flush=True)
    for x in sorted(chosen,key=lambda z:z[0],reverse=True)[:6]:
        print(f'    quality proxy={x[0]:.3f}, triangles={x[2]}, '
              f'neighbor quads={x[3]}, new quads={len(x[5])}, new interior nodes={len(x[4])}', flush=True)
    return chosen, len(tids)



def mesh_2d_counts():
    etypes, tag_lists, _ = gmsh.model.mesh.getElements(2)
    return {int(t): len(v) for t, v in zip(etypes, tag_lists)}


def repair_triangle_patches(threshold=.075, max_extra=2):
    """Phase 1: use existing nodes only, matching successful original run."""
    selected = {}
    original = 0
    for name, surface in outer_surfaces().items():
        picks, ntri = scan_surface(name, surface, threshold, max_extra)
        selected[surface] = picks
        original += ntri
    n_patches = sum(len(v) for v in selected.values())
    print(f'PHASE 1: {n_patches} patches; {2*n_patches}/{original} triangles removable', flush=True)
    for surface, patches in selected.items():
        if not patches:
            continue
        removed = sorted(int(e) for _, affected, _, _, _ in patches for e in affected)
        quads = [q for _, _, _, _, tiling in patches for q in tiling]
        if len(set(removed)) != len(removed):
            raise RuntimeError('Phase 1 overlapping replacement patches')
        gmsh.model.mesh.removeElements(2, int(surface), removed)
        gmsh.model.mesh.addElementsByType(int(surface), 3, [], [int(v) for q in quads for v in q])
        print(f'  surface {surface}: removed {len(removed)} elements; added {len(quads)} quads', flush=True)
    remaining = mesh_2d_counts().get(2, 0)
    if remaining != original - 2*n_patches:
        raise RuntimeError(f'Phase 1 unexpected triangle count {remaining}')
    return n_patches


def repair_triangle_steiner(threshold=.08, max_extra=2, grid=11):
    """Phase 2: add ONLY surface-interior nodes; no interface segmentation."""
    selected = {}
    original = 0
    for name, surface in outer_surfaces().items():
        picks, ntri = pick_patches(name, surface, threshold, max_extra, grid)
        selected[surface] = picks
        original += ntri
    n_patches = sum(len(v) for v in selected.values())
    print(f'PHASE 2: {n_patches} Steiner patches; {2*n_patches}/{original} triangles removable', flush=True)
    next_node = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    for surface, patches in selected.items():
        if not patches:
            continue
        new_tags, new_coords, removed, quad_conn = [], [], [], []
        for score, affected, pair, extra, new_positions, replacements in patches:
            mapping = {}
            for i, xy in enumerate(new_positions):
                mapping[-1-i] = next_node
                new_tags.append(next_node)
                new_coords.extend([float(xy[0]), float(xy[1]), 0.0])
                next_node += 1
            removed.extend(map(int, affected))
            for row in replacements:
                if any(v is None for v in row):
                    raise RuntimeError('Phase 2 patch contains unset vertex placeholder')
                quad_conn.extend(mapping[int(v)] if int(v) < 0 else int(v) for v in row)
        if len(set(removed)) != len(removed):
            raise RuntimeError('Phase 2 overlapping replacement patches')
        gmsh.model.mesh.addNodes(2, int(surface), new_tags, new_coords)
        gmsh.model.mesh.removeElements(2, int(surface), removed)
        gmsh.model.mesh.addElementsByType(int(surface), 3, [], quad_conn)
        print(f'  surface {surface}: added {len(new_tags)} interior nodes; '
              f'removed {len(removed)} elements; added {len(quad_conn)//4} quads', flush=True)
    remaining = mesh_2d_counts().get(2, 0)
    if remaining != original - 2*n_patches:
        raise RuntimeError(f'Phase 2 unexpected triangle count {remaining}')
    return n_patches


def ensure_all_quads(allow_repair=True):
    """Repair residual triangles in a few local passes, then audit all quads."""
    before = mesh_2d_counts()
    print('Before local repair:', before, flush=True)
    if set(before) - {2, 3}:
        raise RuntimeError(f'Unexpected 2-D element types: {before}')
    if before.get(2, 0):
        if not allow_repair:
            raise RuntimeError('Triangles present and automatic repair disabled')
        repair_triangle_patches()
        # A patch may compete with another for a neighboring quad. New quads
        # from the first pass can enable a small follow-up patch. Never loop
        # indefinitely or publish a mesh with residual triangles.
        for attempt in range(1, 4):
            remaining = mesh_2d_counts().get(2, 0)
            if not remaining:
                break
            print(f'STEINER PASS {attempt}: {remaining} triangles', flush=True)
            fixed = repair_triangle_steiner(max_extra=3 if attempt > 1 else 2)
            after_pass = mesh_2d_counts().get(2, 0)
            if after_pass >= remaining or not fixed:
                break
    after = mesh_2d_counts()
    print('After local repair:', after, flush=True)
    if set(after) != {3} or not after[3]:
        raise RuntimeError(f'Not an all-quad mesh: {after}')
    quality = inspect_quad_quality(0.05, 0.10, label='all-quad quality gate')
    if quality['n_failed']:
        raise RuntimeError(f"{quality['n_failed']} quadrilaterals below minSICN acceptance")
    return after, quality



def _ordered_edge_coordinates(all_nodes, all_xyz, conn):
    """Canonical edge geometry independent of Gmsh node numbering."""
    idx = np.argsort(all_nodes)
    tags = all_nodes[idx]
    coords = all_xyz[idx]
    positions = np.searchsorted(tags, conn)
    if np.any(positions >= len(tags)) or not np.array_equal(tags[positions], conn):
        raise RuntimeError('1-D element references a missing mesh node')
    pts = coords[positions].reshape(-1, 2, 3).copy()
    a, b = pts[:, 0, :], pts[:, 1, :]
    swap = ((a[:, 0] > b[:, 0]) |
            ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1])) |
            ((a[:, 0] == b[:, 0]) & (a[:, 1] == b[:, 1]) & (a[:, 2] > b[:, 2])))
    pts[swap] = pts[swap, ::-1, :]
    flat = pts.reshape(-1, 6)
    order = np.lexsort(tuple(flat[:, i] for i in range(5, -1, -1)))
    return flat[order]


def _mesh_edge_keys(a, b):
    a, b = np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)
    if a.size and int(max(a.max(), b.max())) >= 2**32:
        raise RuntimeError('Node tag too large for edge encoding')
    return (np.minimum(a, b) << np.uint64(32)) | np.maximum(a, b)


def _surface_edge_presence(surface, line_nodes):
    types, _, node_arrays = gmsh.model.mesh.getElements(2, int(surface))
    pieces=[]
    for ty, conn in zip(types, node_arrays):
        corners = {2: 3, 3: 4}.get(int(ty))
        if corners is None:
            raise RuntimeError(f'Unexpected element type {ty} on surface {surface}')
        quads = np.asarray(conn, np.uint64).reshape(-1, corners)
        for i in range(corners):
            pieces.append(_mesh_edge_keys(quads[:, i], quads[:, (i+1) % corners]))
    if not pieces:
        raise RuntimeError(f'No 2D cells for surface {surface}')
    known = np.unique(np.concatenate(pieces))
    targets = _mesh_edge_keys(line_nodes[:, 0], line_nodes[:, 1])
    index = np.searchsorted(known, targets)
    present = index < len(known)
    present[present] = known[index[present]] == targets[present]
    return present


def _outer_areas(node_tags, node_xyz, surfaces):
    """Geometric area of water/basement, independent of element and node tags."""
    sort = np.argsort(node_tags)
    sorted_tags, sorted_xyz = node_tags[sort], node_xyz[sort]
    totals={}
    for name, surf in surfaces.items():
        surface_types, _, nodes_by_type = gmsh.model.mesh.getElements(2, int(surf))
        area_total = 0.0
        for typ, flat in zip(surface_types, nodes_by_type):
            nc = {2:3, 3:4}.get(int(typ))
            if nc is None:
                raise RuntimeError(f'Unknown 2D type {typ} in surface {surf}')
            conn = np.asarray(flat,dtype=np.int64).reshape(-1,nc)
            for j in range(0,len(conn),120000):
                subset=conn[j:j+120000]
                indices=np.searchsorted(sorted_tags,subset)
                if np.any(indices>=len(sorted_tags)) or np.any(sorted_tags[indices]!=subset):
                    raise RuntimeError(f'Missing mesh vertex on surface {surf}')
                xy=sorted_xyz[indices,:2]
                xx=xy[:,:,0]; yy=xy[:,:,1]
                a=0.5*np.sum(xx*np.roll(yy,-1,axis=1) -
                             np.roll(xx,-1,axis=1)*yy,axis=1)
                area_total+=float(np.sum(np.abs(a)))
        totals[name]=area_total
    return totals


def mesh_snapshot():
    """Capture geometry, memberships and 1-D/2-D incidence; no stable node IDs assumed."""
    phys = {(int(dim), int(p)): (gmsh.model.getPhysicalName(dim, p),
            tuple(sorted(map(int, gmsh.model.getEntitiesForPhysicalGroup(dim, p)))))
            for dim,p in gmsh.model.getPhysicalGroups()}
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.asarray(nodes, dtype=np.int64)
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    # IMPORTANT: when the 2D mesh is still in memory Gmsh also keeps
    # internal sediment-layer curves. With Mesh.SaveAll=0 MSH2.2 saves
    # ONLY physical 1D curves. Snapshot the SAME physical curves on both sides
    # of a save/reload; otherwise the comparison reports false removals.
    physical_curves=set()
    for dim, physical_tag in gmsh.model.getPhysicalGroups(1):
        physical_curves.update(int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(1, physical_tag))
    pieces=[]
    for curve in sorted(physical_curves):
        types, _, node_lists = gmsh.model.mesh.getElements(1, int(curve))
        if any(int(t)!=1 for t in types):
            raise RuntimeError(f'Unexpected element type on physical 1D curve {curve}: {types}')
        pieces.extend(np.asarray(conn,dtype=np.int64).reshape(-1,2) for conn in node_lists)
    lines = np.concatenate(pieces, axis=0) if pieces else np.empty((0,2), dtype=np.int64)
    geom = _ordered_edge_coordinates(nodes, xyz, lines.ravel()) if len(lines) else np.empty((0,6))
    surfaces = outer_surfaces()
    incidence = {name: _surface_edge_presence(surf, lines) for name, surf in surfaces.items()}
    # Align incidence ordering with geometry-sorted edges. This works even if tags change.
    if len(lines):
        # Compute same sorting order as in _ordered_edge_coordinates from geometric values.
        pos = np.argsort(nodes)
        sorted_tags, sorted_xyz = nodes[pos], xyz[pos]
        lookup = np.searchsorted(sorted_tags, lines.ravel())
        xy = sorted_xyz[lookup].reshape(-1,2,3)
        aa, bb = xy[:,0,:], xy[:,1,:]
        need_swap=((aa[:,0]>bb[:,0])|((aa[:,0]==bb[:,0])&(aa[:,1]>bb[:,1]))|
                   ((aa[:,0]==bb[:,0])&(aa[:,1]==bb[:,1])&(aa[:,2]>bb[:,2])))
        canonical=xy.copy()
        canonical[need_swap]=canonical[need_swap,::-1,:]
        rows=canonical.reshape(-1,6)
        sortidx=np.lexsort(tuple(rows[:,i] for i in range(5,-1,-1)))
        incidence={k:v[sortidx] for k,v in incidence.items()}
    areas=_outer_areas(nodes, xyz, surfaces)
    counts=mesh_2d_counts()
    print(f'  physical groups={len(phys)}; nodes={len(nodes):,}; '
          f'1-D segments={len(lines):,}; 2-D types={counts}', flush=True)
    return {'physical': phys, 'node_xyz': xyz, 'edges_xyz':geom,
            'incidence':incidence, 'counts':counts, 'areas':areas}


def compare_mesh_snapshots(before, after, tol=1e-8):
    """Compare the mesh physically (not by tag): Gmsh can renumber 2D nodes."""
    from scipy.spatial import cKDTree
    if before['physical'] != after['physical']:
        raise RuntimeError('Physical group/entity membership changed')
    print('PASS: physical groups and entity memberships retained', flush=True)
    ea, eb = before['edges_xyz'], after['edges_xyz']
    if ea.shape != eb.shape or not np.allclose(ea, eb, atol=tol, rtol=0):
        raise RuntimeError('1D segment geometry has changed')
    print(f'PASS: {len(ea):,} 1-D segments unchanged geometrically', flush=True)
    for name in before['incidence']:
        if not np.array_equal(before['incidence'][name], after['incidence'][name]):
            raise RuntimeError(f'1D-to-2D surface edge incidence changed on {name}')
        print(f'PASS: {name} 1D-to-2D incidence unchanged', flush=True)
    for name, original in before['areas'].items():
        current=after['areas'][name]
        if not np.isclose(original,current,rtol=1e-8,atol=1e-2):
            raise RuntimeError(f'{name} geometric area changed: {original} -> {current}')
        print(f'PASS: {name} surface area conserved ({current:.6f} m^2; delta={current-original:.6g} m^2)', flush=True)
    a, b = before['node_xyz'], after['node_xyz']
    if len(b) < len(a):
        raise RuntimeError('Original node count decreased')
    tree = cKDTree(b, leafsize=32, balanced_tree=False, compact_nodes=False)
    missing, max_dist = 0, 0.0
    for i in range(0, len(a), 100000):
        distances, _ = tree.query(a[i:i+100000], k=1,
                                  distance_upper_bound=tol, workers=-1)
        missing += int(np.count_nonzero(~np.isfinite(distances)))
        valid=distances[np.isfinite(distances)]
        if len(valid): max_dist=max(max_dist,float(valid.max()))
    print(f'Original coordinates checked by spatial position: {len(a):,}; '
          f'unmatched={missing}; max matched discrepancy={max_dist:.3g} m', flush=True)
    if missing:
        raise RuntimeError(f'{missing} original node positions missing in final mesh')
    print(f'PASS: original geometry preserved; {len(b)-len(a):,} additional nodes', flush=True)
    return True


def audit_global_quad_edges():
    """All internal edges shared twice, only four physical outer sides shared once.

    This is independent of Gmsh geometric curve data, which may be absent
    from a reloaded MSH2.2 when Mesh.SaveAll=0. Thus it checks the ungrouped
    sediment/basement interface too, not merely the physical water interface.
    """
    types, _, nodes_by_type = gmsh.model.mesh.getElements(2)
    if set(map(int,types))!={3}:
        raise RuntimeError(f'Cannot audit quad manifold with types {types}')
    all_keys=[]
    for data in nodes_by_type:
        conn=np.asarray(data,dtype=np.uint64).reshape(-1,4)
        for j in range(4):
            all_keys.append(_mesh_edge_keys(conn[:,j],conn[:,(j+1)%4]))
    all_keys=np.concatenate(all_keys)
    all_keys.sort()
    first=np.r_[0, np.flatnonzero(all_keys[1:]!=all_keys[:-1])+1]
    counts=np.diff(np.r_[first,len(all_keys)])
    if np.any(counts>2):
        raise RuntimeError(f'{np.count_nonzero(counts>2)} 2D edges used by >2 quads')
    single=np.asarray(all_keys[first[counts==1]],dtype=np.uint64)
    del all_keys, first, counts
    # Compare all one-sided edges to physical lateral/top/bottom segments.
    outer_curves=set()
    for dim,phys in gmsh.model.getPhysicalGroups(1):
        if gmsh.model.getPhysicalName(dim,phys) in ('xmin','xmax','zmin','zmax'):
            outer_curves.update(int(v) for v in gmsh.model.getEntitiesForPhysicalGroup(1,phys))
    if not outer_curves:
        raise RuntimeError('Missing physical outer boundaries xmin/xmax/zmin/zmax')
    expected=[]
    for curve in sorted(outer_curves):
        types,_,conn=gmsh.model.mesh.getElements(1,curve)
        for typ,flat in zip(types,conn):
            if int(typ)!=1:
                raise RuntimeError('Nonlinear 1D element on outer boundary')
            edge=np.asarray(flat,dtype=np.uint64).reshape(-1,2)
            expected.append(_mesh_edge_keys(edge[:,0],edge[:,1]))
    boundary=np.concatenate(expected) if expected else np.empty(0,dtype=np.uint64)
    if not len(boundary) or len(np.unique(boundary))!=len(boundary):
        raise RuntimeError('Missing/duplicated physical outer side segments')
    if len(single)!=len(boundary) or not np.array_equal(np.sort(single),np.sort(boundary)):
        raise RuntimeError(f'Quad boundary mismatch: one-sided edges={len(single)}, '
                           f'physical outer segments={len(boundary)}')
    print(f'PASS: global 2D manifold and boundary closure; '
          f'{len(single):,} one-sided edges match physical outer boundary',flush=True)
    return len(single)


def validate_loaded_mesh(reference=None, threshold=.05):
    """Check a loaded Gmsh model; optionally compare to a pre-repair .msh."""
    now=mesh_snapshot()
    counts=now['counts']
    if set(counts) != {3}:
        raise RuntimeError(f'Not all-quadrilateral: {counts}')
    if not now['physical']:
        raise RuntimeError('No physical groups present')
    report=inspect_quad_quality(threshold, 0.1, label='full .msh quality audit')
    if report['n_failed']:
        raise RuntimeError(f"{report['n_failed']} quads below quality threshold")
    if reference is not None:
        gmsh.clear()
        gmsh.open(str(reference))
        old=mesh_snapshot()
        compare_mesh_snapshots(old,now)
    return now, report


def material_data_from_physical_groups():
    """Rebuild EXACT generator material constants for an existing corrected .msh."""
    names={gmsh.model.getPhysicalName(d,p) for d,p in gmsh.model.getPhysicalGroups(2)}
    if 'socle_1' not in names or 'socle_3' not in names:
        raise RuntimeError('Missing socle_1 or socle_3 material')
    import re
    sed=sorted(names-{'socle_1','socle_3'})
    if not sed or any(not re.fullmatch(r'sed_vs_\d+(?:\.\d+)?', n) for n in sed):
        raise RuntimeError(f'Unexpected sediment physical groups: {sed[:10]}')
    return {'acoustic_2d': {'socle_1': {'rho':1000,'vp':1500}},
            'viscoelastic_2d': {
            **{n:{'rho':1800,'vp':1800,'vs':float(n.split('_')[-1]),
                 'q_kappa':100,'q_mu':50,'f_att':50.0} for n in sed},
            'socle_3':{'rho':2400,'vp':4000,'vs':1600,
                       'q_kappa':250,'q_mu':150,'f_att':50.0}}}




def final_meshing(
    filename, use_global_quad_subdivision=False, water_surface=None,
    sediment_surfaces=(), rock_surface=None, top_interface_curves=(),
    bottom_interface_curves=(), unstructured_algorithm=5,
):
    """Hybrid Gmsh + automatic local repair -> one validated MSH2 binary file.

    This function is the ONLY generator finalization routine. It does not
    regenerate the mesh to repair residual triangles, and does not touch
    boundaries or layer geometry.
    """
    filename=Path(filename)
    if water_surface is None or rock_surface is None:
        raise ValueError('Missing outer surfaces')
    if use_global_quad_subdivision:
        raise ValueError('Global quad subdivision incompatible with transfinite sediments')
    if unstructured_algorithm not in (5,6,8):
        raise ValueError('outer algorithm must be 5, 6, or 8')
    opts={'Geometry.MatchMeshTolerance':1e-9,'Geometry.Tolerance':1e-9,
          'Mesh.Algorithm':int(unstructured_algorithm),
          'Mesh.RecombineAll':0,'Mesh.RecombinationAlgorithm':1,
          'Mesh.RecombineMinimumQuality':0.05,
          'Mesh.SubdivisionAlgorithm':0,'Mesh.ElementOrder':1,
          'Mesh.SecondOrderLinear':0,'Mesh.Smoothing':10,
          'Mesh.MshFileVersion':2.2,'Mesh.Binary':1,
          'Mesh.RecombineOptimizeTopology':5,
          'Mesh.CharacteristicLengthFactor':1,'Mesh.SaveAll':0}
    for name,val in opts.items(): gmsh.option.setNumber(name,val)
    gmsh.model.mesh.setAlgorithm(2,int(water_surface),int(unstructured_algorithm))
    gmsh.model.mesh.setAlgorithm(2,int(rock_surface),int(unstructured_algorithm))
    gmsh.model.mesh.setRecombine(2,int(water_surface))
    gmsh.model.mesh.setRecombine(2,int(rock_surface))
    print('\nHYBRID: sediment transfinite / outer Delaunay + standard Blossom',flush=True)
    _assert_interface_curve_sharing(water_surface,rock_surface,
                                   top_interface_curves,bottom_interface_curves)
    gmsh.model.mesh.generate(2)
    report_mesh_element_distribution(water_surface,sediment_surfaces,rock_surface)
    _assert_interface_quad_edges(water_surface,rock_surface,
                                 top_interface_curves,bottom_interface_curves)
    # Preserve a restartable raw checkpoint. If a local repair fails, use
    # repair_mesh.py --repair on this file instead of regenerating 2.6M cells.
    filename.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=filename.with_name(filename.stem+'.unrepaired.msh')
    if checkpoint.exists():
        raise RuntimeError(f'Raw mesh checkpoint already exists: {checkpoint}. '
                           'Use repair_mesh.py --repair or archive it first.')
    gmsh.write(str(checkpoint))
    print(f'RESTART CHECKPOINT: {checkpoint}', flush=True)
    before=mesh_snapshot()
    ensure_all_quads(allow_repair=True)
    _assert_interface_quad_edges(water_surface,rock_surface,
                                 top_interface_curves,bottom_interface_curves)
    print('\nVALIDATE AFTER REPAIR (in-memory)', flush=True)
    after=mesh_snapshot()
    compare_mesh_snapshots(before,after)
    # Never overwrite an existing canonical .msh. Save a unique temporary file.
    if filename.exists():
        raise RuntimeError(f'Validated output already exists: {filename}; no overwrite')
    filename.parent.mkdir(parents=True,exist_ok=True)
    temp=filename.with_name(filename.stem+'.incomplete.msh')
    if temp.exists():
        raise RuntimeError(f'Stale incomplete mesh exists: {temp}; inspect it first')
    gmsh.write(str(temp))
    print('\nVALIDATE SAVED MESH',flush=True)
    gmsh.clear()
    gmsh.open(str(temp))
    reopened=mesh_snapshot()
    compare_mesh_snapshots(before,reopened)
    count=mesh_2d_counts()
    if set(count)!={3}:
        raise RuntimeError(f'Reopened mesh not all quad: {count}')
    inspect=inspect_quad_quality(0.05,0.1,label='reopened file quality audit')
    audit_global_quad_edges()
    if inspect['n_failed']:
        raise RuntimeError('Saved file has poor quads')
    # Reopened geometry/entity tags are not required to match original node tags.
    temp.replace(filename)
    checkpoint.unlink()  # only after successful re-open and global audit
    print(f'\nPASS: ONE VALIDATED MSH: {filename}',flush=True)
    return filename
