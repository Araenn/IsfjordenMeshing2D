"""Gmsh helpers used by isfjorden_msh.py only."""

import gmsh
import numpy as np

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
    top_interface_groups = {}
    for ix in range(len(x) - 1):
        for iz in range(n_vertical):
            loop = factory.addCurveLoop([
                horizontal[iz][ix], vertical[ix + 1][iz],
                -horizontal[iz + 1][ix], -vertical[ix][iz],
            ])
            surface = factory.addPlaneSurface([loop])
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
    return materials, groups, top_interface_groups, boundary_curves



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

def final_meshing(
    filename,
    use_global_quad_subdivision=True,
    water_surface=None,
    sediment_surfaces=(),
    rock_surface=None,
):
    """Generate a first-order quadrilateral mesh in Gmsh 2.2 format.

    ``use_global_quad_subdivision=True`` keeps the previously validated robust
    full-quad path. Set it to False only for dedicated element-count tests:
    some surfaces can then retain triangles and the final validation will stop.
    """
    options = {
        "Geometry.MatchMeshTolerance": 1e-9,
        "Geometry.Tolerance": 1e-9,
        "Mesh.Algorithm": 8,
        "Mesh.RecombineAll": 1,
        "Mesh.RecombinationAlgorithm": 1,
        "Mesh.RecombineMinimumQuality": 0.0,
        "Mesh.SubdivisionAlgorithm": 1 if use_global_quad_subdivision else 0,
        "Mesh.ElementOrder": 1,
        "Mesh.SecondOrderLinear": 0,
        "Mesh.Smoothing": 10,
        "Mesh.MshFileVersion": 2.2,
        "Mesh.RecombineOptimizeTopology": 5,
        "Mesh.CharacteristicLengthFactor": 1,
        "Mesh.SaveAll": 0,
    }
    for name, value in options.items():
        gmsh.option.setNumber(name, value)

    print(
        "Global quad subdivision: "
        + ("enabled (robust reference mode)" if use_global_quad_subdivision
           else "disabled (optimization test mode)")
    )

    gmsh.model.mesh.generate(2)
    types, element_tags, _ = gmsh.model.mesh.getElements(2)
    for element_type, tags in zip(types, element_tags):
        print(f"Element type {element_type}: {len(tags)} elements")

    if water_surface is not None and rock_surface is not None:
        report_mesh_element_distribution(
            water_surface=water_surface,
            sediment_surfaces=sediment_surfaces,
            rock_surface=rock_surface,
        )
    if len(types) == 0 or any(int(element_type) != 3 for element_type in types):
        raise RuntimeError(
            "Mesh must contain only first-order quadrilateral elements (type 3). "
            "If global quad subdivision is disabled, re-enable it or improve "
            "recombination locally."
        )
    gmsh.write(str(filename))

