"""Gmsh helpers used by isfjordenmsh.py only."""

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
    """Use the sediment interface curves directly for water or basement."""
    left = factory.addPoint(float(x[0]), float(outer_z), 0, float(lc))
    right = factory.addPoint(float(x[-1]), float(outer_z), 0, float(lc))
    if above:
        edges = [
            factory.addLine(left, right),
            factory.addLine(right, interface_points[-1]),
            *[-edge for edge in reversed(interface_edges)],
            factory.addLine(interface_points[0], left),
        ]
    else:
        edges = [
            *interface_edges,
            factory.addLine(interface_points[-1], right),
            factory.addLine(right, left),
            factory.addLine(left, interface_points[0]),
        ]
    return factory.addPlaneSurface([factory.addCurveLoop(edges)])


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

    materials = {
        name: {
            "rho": 1800, "vp": 1800, "vs": float(name.removeprefix("sed_vs_")),
            "q_kappa": 100, "q_mu": 50, "f_att": f_att,
        }
        for name in groups
    }
    return materials, groups


def add_physical_group(surfaces, tag, name):
    gmsh.model.addPhysicalGroup(2, surfaces, tag)
    gmsh.model.setPhysicalName(2, tag, name)


def lc_from_vs(vs, fmax=50.0, nppw=6, lc_min=1.0, lc_max=20.0):
    return np.clip(np.asarray(vs) / (fmax * nppw), lc_min, lc_max)


def final_meshing(filename):
    """Generate a first-order quadrilateral mesh in Gmsh 2.2 format."""
    options = {
        "Geometry.MatchMeshTolerance": 1e-9,
        "Geometry.Tolerance": 1e-9,
        "Mesh.Algorithm": 8,
        "Mesh.RecombineAll": 1,
        "Mesh.RecombinationAlgorithm": 1,
        # Subdivide any remaining triangles into quads while preserving interfaces.
        "Mesh.SubdivisionAlgorithm": 1,
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
    gmsh.model.mesh.generate(2)
    types, element_tags, _ = gmsh.model.mesh.getElements(2)
    for element_type, tags in zip(types, element_tags):
        print(f"Element type {element_type}: {len(tags)} elements")
    if len(types) == 0 or any(int(element_type) != 3 for element_type in types):
        raise RuntimeError("Mesh must contain only first-order quadrilateral elements (type 3).")
    gmsh.write(str(filename))
