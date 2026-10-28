import h5py
import argparse
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations

def test_end_of_section(line, test, binary: bool):
    line = (str(line, 'utf-8') if binary else line).rstrip()
    if line != test:
        raise ValueError(f"Input file '{test.replace('End', '')}' section does not end with '{test}'")


@dataclass
class Physic:
    dim: int
    name: str


@dataclass
class Entity:
    tags: np.array
    min_coords: np.array
    max_coords: np.array
    bounding: np.array


def parse_physical_names(file, binary, cache=None):
    """Parse Gmsh PhysicalNames and normalize quoted names.

    In Gmsh 2.2, names are written for example as::

        2 1 "socle_1"

    The previous parser kept the quote characters, producing names such as
    ``"socle_1"`` instead of ``socle_1``.
    """
    if cache is None:
        cache = dict()

    for _ in range(int(file.readline().rstrip())):
        raw = file.readline()
        if binary:
            raw = raw.decode('utf-8')
        raw = raw.rstrip()

        # split only twice so physical names containing spaces are preserved
        dim, index, name = raw.split(maxsplit=2)
        name = name.strip()
        if len(name) >= 2 and name[0] == name[-1] and name[0] in ('"', "'"):
            name = name[1:-1]

        cache[int(index)] = Physic(int(dim), name)

    test_end_of_section(file.readline(), '$EndPhysicalNames', binary)
    return cache


def parse_entities_ascii(file, cache=None):
    num_points, num_curves, num_surfaces, num_volumes = map(int, file.readline().rstrip().split())
    if cache is None:
        cache = [dict(), dict(), dict(), dict()]
    for i in range(num_points):
        line = file.readline().rstrip().split()
        point_tag = int(line[0])
        coord = np.array(line[1:4], dtype=np.float64)
        num_physical_tag = int(line[4])
        tags = np.array(line[5:], dtype=np.int32)
        cache[0][point_tag] = Entity(tags, coord, coord, [])
    for i in range(num_curves):
        line = file.readline().rstrip().split()
        curve_tag = int(line[0])
        min_of_coord = np.array(line[1:4], dtype=np.float64)
        max_of_coord = np.array(line[4:7], dtype=np.float64)
        num_physical_tag = int(line[7])
        tags = np.array(line[8:8+num_physical_tag], dtype=np.int32)
        num_bounding_points = int(line[8+num_physical_tag])
        point_tags = np.array(line[9+num_physical_tag:], dtype=np.int32)
        cache[1][curve_tag] = Entity(tags, min_of_coord, max_of_coord, point_tags)
    for i in range(num_surfaces):
        line = file.readline().rstrip().split()
        surface_tag = int(line[0])
        min_of_coord = np.array(line[1:4], dtype=np.float64)
        max_of_coord = np.array(line[4:7], dtype=np.float64)
        num_physical_tag = int(line[7])
        tags = np.array(line[8:8+num_physical_tag], dtype=np.int32)
        num_bounding_curves = int(line[8+num_physical_tag])
        curve_tags = np.array(line[9+num_physical_tag:], dtype=np.int32)
        cache[2][surface_tag] = Entity(tags, min_of_coord, max_of_coord, curve_tags)
    for i in range(num_volumes):
        line = file.readline().rstrip().split()
        volume_tag = int(line[0])
        min_of_coord = np.array(line[1:4], dtype=np.float64)
        max_of_coord = np.array(line[4:7], dtype=np.float64)
        num_physical_tag = int(line[7])
        tags = np.array(line[8:8+num_physical_tag], dtype=np.int32)
        num_bounding_surfaces = int(line[8+num_physical_tag])
        surface_tags = np.array(line[9+num_physical_tag:], dtype=np.int32)
        cache[3][volume_tag] = Entity(tags, min_of_coord, max_of_coord, surface_tags)
    test_end_of_section(file.readline(), '$EndEntities', False)
    return cache


def parse_entities_binary(file, size_of_size_t, endian, cache=None):
    if cache is None:
        cache = [dict(), dict(), dict(), dict()]
    num_points = int.from_bytes(file.read(size_of_size_t), endian)
    num_curves = int.from_bytes(file.read(size_of_size_t), endian)
    num_surfaces = int.from_bytes(file.read(size_of_size_t), endian)
    num_volumes = int.from_bytes(file.read(size_of_size_t), endian)
    e = "<" if endian == "little" else ">"
    for i in range(num_points):
        point_tag = int.from_bytes(file.read(4), endian)
        coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        num_physical_tag = int.from_bytes(file.read(size_of_size_t), endian)
        tags = np.frombuffer(file.read(4*num_physical_tag), dtype=f"{e}i{4}")
        cache[0][point_tag] = Entity(tags, coord, coord, [])
    for i in range(num_curves):
        curve_tag = int.from_bytes(file.read(4), endian)
        min_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        max_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        num_physical_tag = int.from_bytes(file.read(size_of_size_t), endian)
        tags = np.frombuffer(file.read(4*num_physical_tag), dtype=f"{e}i{4}")
        num_bounding_points = int.from_bytes(file.read(size_of_size_t), endian)
        point_tags = np.frombuffer(file.read(4*num_bounding_points), dtype=f"{e}i{4}")
        cache[1][curve_tag] = Entity(tags, min_of_coord, max_of_coord, point_tags)
    for i in range(num_surfaces):
        surface_tag = int.from_bytes(file.read(4), endian)
        min_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        max_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        num_physical_tag = int.from_bytes(file.read(size_of_size_t), endian)
        tags = np.frombuffer(file.read(4*num_physical_tag), dtype=f"{e}i{4}")
        num_bounding_curves = int.from_bytes(file.read(size_of_size_t), endian)
        curve_tags = np.frombuffer(file.read(4*num_bounding_curves), dtype=f"{e}i{4}")
        cache[2][surface_tag] = Entity(tags, min_of_coord, max_of_coord, curve_tags)
    for i in range(num_volumes):
        volume_tag = int.from_bytes(file.read(4), endian)
        min_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        max_of_coord = np.frombuffer(file.read(3*size_of_size_t), dtype=f"{e}f{size_of_size_t}")
        num_physical_tag = int.from_bytes(file.read(size_of_size_t), endian)
        tags = np.frombuffer(file.read(4*num_physical_tag), dtype=f"{e}i{4}")
        num_bounding_surfaces = int.from_bytes(file.read(size_of_size_t), endian)
        surface_tags = np.frombuffer(file.read(4*num_bounding_surfaces), dtype=f"{e}i{4}")
        cache[3][volume_tag] = Entity(tags, min_of_coord, max_of_coord, surface_tags)
    file.read(1)  # should be the b'\n'
    test_end_of_section(file.readline(), '$EndEntities', True)
    return cache


def parse_partitioned_entities():
    """numPartitions(size_t)
    numGhostEntities(size_t)
      ghostEntityTag(int) partition(int)
      ...
    numPoints(size_t) numCurves(size_t)
      numSurfaces(size_t) numVolumes(size_t)
    pointTag(int) parentDim(int) parentTag(int)
      numPartitions(size_t) partitionTag(int) ...
      X(double) Y(double) Z(double)
      numPhysicalTags(size_t) physicalTag(int) ...
    ...
    curveTag(int) parentDim(int) parentTag(int)
      numPartitions(size_t) partitionTag(int) ...
      minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundingPoints(size_t) pointTag(int) ...
    ...
    surfaceTag(int) parentDim(int) parentTag(int)
      numPartitions(size_t) partitionTag(int) ...
      minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundingCurves(size_t) curveTag(int) ...
    ...
    volumeTag(int) parentDim(int) parentTag(int)
      numPartitions(size_t) partitionTag(int) ...
      minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundingSurfaces(size_t) surfaceTag(int) ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndPartitionedEntities', True)


def parse_nodes_2_2_ascii(file, cache=None):
    num_nodes = int(file.readline().rstrip())
    # for i in range(num_nodes):
    #     tag, coord = file.readline().rstrip().split(' ', 1)
    #     coord = np.array(coord.split(), dtype=np.float64)
    lines = [file.readline().rstrip().split() for _ in range(num_nodes)]
    test_end_of_section(file.readline(), '$EndNodes', False)
    if num_nodes == 0:
        return cache
    df = pd.DataFrame(lines, columns=['tag', 'x', 'y', 'z'])
    df = df.astype({'tag': 'int32', 'x': float, 'y': float, 'z': float})
    df.set_index('tag', inplace=True)
    if cache is None:
        return df
    return pd.concat([df, cache], axis=0)


def parse_nodes_2_2_binary(file, size_of_size_t, endian, cache=None):
    num_nodes = int(file.readline().rstrip())
    e = "<" if endian == "little" else ">"
    arr = np.frombuffer(file.read((4+3*size_of_size_t)*num_nodes), dtype=','.join(['<i4'] + 3*[f"{e}f{size_of_size_t}"]))
    df = pd.DataFrame.from_dict({'tag': arr['f0'], 'x': arr['f1'], 'y': arr['f2'], 'z': arr['f3']})
    df.set_index('tag', inplace=True)
    file.read(1)  # should be the b'\n'
    test_end_of_section(file.readline(), '$EndNodes', True)
    if cache is None:
        return df
    return pd.concat([cache, df], axis=0)


def parse_nodes_4_1_ascii(file, cache=None):
    num_entity_blocks, num_nodes, min_node_tag, max_node_tag = map(int, file.readline().rstrip().split())
    df = list()
    for i in range(num_entity_blocks):
        entity_dim, entity_tag, parametric, num_nodes_in_block = map(int, file.readline().rstrip().split())
        if parametric == 1:
            raise NotImplementedError("Parametric nodes are not yet implemented.")
        node_tags = np.array([file.readline().rstrip() for _ in range(num_nodes_in_block)], dtype=int)
        coord = np.array([file.readline().rstrip().split() for _ in range(num_nodes_in_block)], dtype=np.float64)
        df.append(pd.DataFrame(coord, columns=['x', 'y', 'z'], index=node_tags).rename_axis('tag'))
        df[-1]['entity_dim'] = entity_dim
        df[-1]['entity_tag'] = entity_tag
    test_end_of_section(file.readline(), '$EndNodes', False)
    if cache is None:
        return pd.concat(df, axis=0)
    return pd.concat([cache] + df, axis=0)


def parse_nodes_4_1_binary(file, size_of_size_t, endian, cache=None):
    num_entity_blocks = int.from_bytes(file.read(size_of_size_t), endian)
    num_nodes = int.from_bytes(file.read(size_of_size_t), endian)
    min_node_tag = int.from_bytes(file.read(size_of_size_t), endian)
    max_node_tag = int.from_bytes(file.read(size_of_size_t), endian)
    e = "<" if endian == "little" else ">"
    df = list()
    for i in range(num_entity_blocks):
        entity_dim = int.from_bytes(file.read(4), endian)
        entity_tag = int.from_bytes(file.read(4), endian)
        parametric = int.from_bytes(file.read(4), endian) == 1
        if parametric:
            raise NotImplementedError("Parametric nodes are not yet implemented.")
        num_nodes_in_block = int.from_bytes(file.read(size_of_size_t), endian)
        node_tags = np.frombuffer(file.read(num_nodes_in_block*size_of_size_t), f"{e}i{size_of_size_t}")
        coord = np.frombuffer(file.read(num_nodes_in_block*size_of_size_t*3), f"{e}f{size_of_size_t}").reshape(-1, 3)
        df.append(pd.DataFrame(coord, columns=['x', 'y', 'z'], index=node_tags).rename_axis('tag'))
        df[-1]['entity_dim'] = entity_dim
        df[-1]['entity_tag'] = entity_tag
    file.read(1)  # should be the b'\n'
    test_end_of_section(file.readline(), '$EndNodes', True)
    if cache is None:
        return pd.concat(df, axis=0)
    return pd.concat([cache] + df, axis=0)



def parse_elements_2_2_ascii(file, cache=None):
    num_elements = int(file.readline().rstrip())
    df = list()
    for _ in range(num_elements):
        line = file.readline().rstrip().split()
        index, element_type, number_of_tag = map(int, line[:3])
        assert number_of_tag >= 2, "Standard gmsh file requires number of tag to be at least 2."
        element_type = parse_elements_4_1_binary.element_types[element_type]
        entity_tag = int(line[3])
        model_tag = int(line[4])
        element_tag = np.array(line[5:5+number_of_tag-2], dtype=int)
        node_index = np.array(line[5+number_of_tag-2:], dtype=int)
        df.append([index, element_type, entity_tag, model_tag, element_tag, node_index])
    test_end_of_section(file.readline(), '$EndElements', False)
    df = pd.DataFrame(df, columns=['tag', 'element_type', 'entity_tag', 'model_tag', 'element_tag', 'node_index']
                      ).set_index('tag')
    if cache is None:
        return df
    return pd.concat([cache, df], axis=0)


def parse_elements_2_2_binary(file, size_of_size_t, endian, cache=None):
    num_elements = int(file.readline().rstrip())
    df = list()
    e = "<" if endian == "little" else ">"
    for _ in range(num_elements):
        element_type = parse_elements_4_1_binary.element_types[int.from_bytes(file.read(4), endian)]
        element_size = parse_elements_4_1_binary.element_size.get(element_type, 0)
        if element_size == 0:
            raise NotImplementedError(element_type + " is not yet implemented.")
        num_elements_in_block = int.from_bytes(file.read(4), endian)
        num_tags = int.from_bytes(file.read(4), endian)
        assert num_tags >= 2, "Standard gmsh file requires number of tag to be at least 2."
        for _ in range(num_elements_in_block):  # mostly size 1 loop, so it's faster to build each row of the dataframe,
            index = int.from_bytes(file.read(4), endian)  # than reading full blocks and building small dataframes
            entity_tag = int.from_bytes(file.read(4), endian)
            model_tag = int.from_bytes(file.read(4), endian)
            tags = np.frombuffer(file.read(4*(num_tags-2)), f"{e}i{4}")
            node_index = np.frombuffer(file.read(4*element_size), f"{e}i{4}")
            df.append([index, element_type, entity_tag, model_tag, tags, node_index])
    file.read(1)  # should be '\n'
    test_end_of_section(file.readline(), '$EndElements', True)
    df = pd.DataFrame(df, columns=['tag', 'element_type', 'entity_tag', 'model_tag', 'element_tag', 'node_index']
                      ).set_index('tag')
    if cache is None:
        return df
    return pd.concat([cache, df], axis=0)


def parse_elements_4_1_ascii(file, cache=None):
    num_entity_blocks, num_elements, min_elements_tag, max_elements_tag = map(int, file.readline().rstrip().split())
    df = list()
    for i in range(num_entity_blocks):
        entity_dim, entity_tag, element_type, num_nodes_in_block = map(int, file.readline().rstrip().split())
        element_type = parse_elements_4_1_binary.element_types[element_type]
        element_size = parse_elements_4_1_binary.element_size.get(element_type, 0)
        if element_size == 0:
            raise NotImplementedError(element_type + " is not yet implemented.")
        element_node_tags = np.array([file.readline().rstrip().split() for _ in range(num_nodes_in_block)], dtype=int)
        df.append(pd.DataFrame(index=element_node_tags[:, 0]))
        df[-1]['node_index'] = list(element_node_tags[:, 1:])
        df[-1]['entity_dim'] = entity_dim
        df[-1]['entity_tag'] = entity_tag
        df[-1]['element_type'] = element_type
    test_end_of_section(file.readline(), '$EndElements', False)
    if cache is None:
        return pd.concat(df, axis=0)
    return pd.concat([cache] + df, axis=0)


def parse_elements_4_1_binary(file, size_of_size_t, endian, cache=None):
    df = list()
    num_entity_blocks = int.from_bytes(file.read(size_of_size_t), endian)
    num_elements = int.from_bytes(file.read(size_of_size_t), endian)
    min_elements_tag = int.from_bytes(file.read(size_of_size_t), endian)
    max_elements_tag = int.from_bytes(file.read(size_of_size_t), endian)
    e = "<" if endian == "little" else ">"
    for i in range(num_entity_blocks):
        entity_dim = int.from_bytes(file.read(4), endian)
        entity_tag = int.from_bytes(file.read(4), endian)
        element_type = parse_elements_4_1_binary.element_types[int.from_bytes(file.read(4), endian)]
        num_nodes_in_block = int.from_bytes(file.read(size_of_size_t), endian)
        element_size = parse_elements_4_1_binary.element_size.get(element_type, 0)
        if element_size == 0:
            raise NotImplementedError(element_type + " is not yet implemented.")
        element_node_tags = np.frombuffer(file.read(num_nodes_in_block * size_of_size_t*(1+element_size)),
                                          f"{e}i{size_of_size_t}").reshape(-1, 1+element_size)
        df.append(pd.DataFrame(index=element_node_tags[:, 0]))
        df[-1]['node_index'] = list(element_node_tags[:, 1:])
        df[-1]['entity_dim'] = entity_dim
        df[-1]['entity_tag'] = entity_tag
        df[-1]['element_type'] = element_type
    file.read(1)  # should be the b'\n'
    test_end_of_section(file.readline(), '$EndElements', True)
    if cache is None:
        return pd.concat(df, axis=0)
    return pd.concat([cache] + df, axis=0)


parse_elements_4_1_binary.element_types = [None,  # types and order from GmshDefines.h
            'LIN_2', 'TRI_3', 'QUA_4', 'TET_4', 'HEX_8', 'PRI_6', 'PYR_5',
            'LIN_3', 'TRI_6', 'QUA_9', 'TET_10', 'HEX_27', 'PRI_18', 'PYR_14',
            'PNT', 'QUA_8', 'HEX_20', 'PRI_15', 'PYR_13',
            'TRI_9', 'TRI_10', 'TRI_12', 'TRI_15', 'TRI_15I', 'TRI_21',
            'LIN_4', 'LIN_5', 'LIN_6',
            'TET_20', 'TET_35', 'TET_56', 'TET_22', 'TET_28',
            'POLYG_', 'POLYH_',
            'QUA_16', 'QUA_25', 'QUA_36', 'QUA_12', 'QUA_16I', 'QUA_20',
            'TRI_28', 'TRI_36', 'TRI_45', 'TRI_55', 'TRI_66',
            'QUA_49', 'QUA_64', 'QUA_81', 'QUA_100', 'QUA_121',
            'TRI_18', 'TRI_21I', 'TRI_24', 'TRI_27', 'TRI_30',
            'QUA_24', 'QUA_28', 'QUA_32', 'QUA_36I', 'QUA_40',
            'LIN_7', 'LIN_8', 'LIN_9', 'LIN_10', 'LIN_11',
            'LIN_B', 'TRI_B', 'POLYG_B', 'LIN_C',
            'TET_84', 'TET_120', 'TET_165', 'TET_220', 'TET_286',  # complete tetra order 6 -> 10
            'TET_34', 'TET_40', 'TET_46', 'TET_52', 'TET_58',  # incomplete tetra order 6 -> 10
            'LIN_1', 'TRI_1', 'QUA_1', 'TET_1', 'HEX_1', 'PRI_1',
            'PRI_40', 'PRI_75',
            'HEX_64', 'HEX_125', 'HEX_216', 'HEX_343', 'HEX_512', 'HEX_729', 'HEX_1000',  # complete hex order 3 -> 9
            'HEX_32', 'HEX_44', 'HEX_56', 'HEX_68', 'HEX_80', 'HEX_92', 'HEX_104',  # incomplete hex order 3 -> 9
            'PRI_126', 'PRI_196', 'PRI_288', 'PRI_405', 'PRI_550',  # complete prism order 3 -> 9
            'PRI_24', 'PRI_33', 'PRI_42', 'PRI_51', 'PRI_60', 'PRI_69', 'PRI_78',  # incomplete prism order 3 -> 9
            'PYR_30', 'PYR_55', 'PYR_91', 'PYR_140', 'PYR_204', 'PYR_285', 'PYR_385',  # complete pyramid order 3 -> 9
            'PYR_21', 'PYR_29', 'PYR_37', 'PYR_45', 'PYR_53', 'PYR_61', 'PYR_69',  # incomplete pyramid order 3 -> 9
            'PYR_1', 'PNT_SUB', 'LIN_SUB', 'TRI_SUB', 'TET_SUB', 'TET_16', 'TRI_MINI', 'TET_MINI',  # additional types
            'TRIH_4']  # Tri hedron -- One quad face and two triangular faces -- made for connecting hexes and tets
parse_elements_4_1_binary.element_size = {'QUA_4': 4, 'QUA_8': 8, 'QUA_9': 9, 'Hex_8': 8, 'Hex_20': 20, 'Hex_27': 27}


def parse_periodic():
    """numPeriodicLinks(size_t)
    entityDim(int) entityTag(int) entityTagMaster(int)
    numAffine(size_t) value(double) ...
    numCorrespondingNodes(size_t)
      nodeTag(size_t) nodeTagMaster(size_t)
      ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndPeriodic', True)

def parse_ghost_elements():
    """numGhostElements(size_t)
    elementTag(size_t) partitionTag(int)
      numGhostPartitions(size_t) ghostPartitionTag(int) ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndGhostElements', True)

def parse_parametrizations():
    """numCurveParam(size_t) numSurfaceParam(size_t)
    curveTag(int) numNodes(size_t)
      nodeX(double) nodeY(double) nodeZ(double) nodeU(double)
      ...
    ...
    surfaceTag(int) numNodes(size_t) numTriangles(size_t)
      nodeX(double) nodeY(double) nodeZ(double)
        nodeU(double) nodeV(double)
        curvMaxX(double) curvMaxY(double) curvMaxZ(double)
        curvMinX(double) curvMinY(double) curvMinZ(double)
      ...
      nodeIndex1(int) nodeIndex2(int) nodeIndex3(int)
      ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndParametrizations', True)

def parse_node_data():
    """numStringTags(ASCII int)
    stringTag(string) ...
    numRealTags(ASCII int)
    realTag(ASCII double) ...
    numIntegerTags(ASCII int)
    integerTag(ASCII int) ...
    nodeTag(int) value(double) ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndNodeData', True)

def parse_element_data():
    """numStringTags(ASCII int)
    stringTag(string) ...
    numRealTags(ASCII int)
    realTag(ASCII double) ...
    numIntegerTags(ASCII int)
    integerTag(ASCII int) ...
    elementTag(int) value(double) ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndElementData', True)

def parse_element_node_data():
    """numStringTags(ASCII int)
    stringTag(string) ...
    numRealTags(ASCII int)
    realTag(ASCII double) ...
    numIntegerTags(ASCII int)
    integerTag(ASCII int) ...
    elementTag(int) numNodesPerElement(int) value(double) ...
    ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndElementNodeData', True)

def parse_interpolation_scheme():
    """name(string)
    numElementTopologies(ASCII int)
    elementTopology
    numInterpolationMatrices(ASCII int)
    numRows(ASCII int) numColumns(ASCII int) value(ASCII double) ..."""
    raise NotImplementedError()
    test_end_of_section(f.readline(), '$EndInterpolationScheme', True)



def _quad_signed_area2(connect, coord):
    """Twice the signed polygon area for Quad4 connectivity."""
    connect = np.asarray(connect, dtype=np.int64)
    x = np.asarray(coord[0], dtype=np.float64)[connect - 1]
    y = np.asarray(coord[1], dtype=np.float64)[connect - 1]
    return np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)


def orient_connectivity_for_specfem(connect, dim, coord=None):
    """Return positively oriented Quad4 connectivity.

    Gmsh guarantees a cyclic Quad4 ordering but the first corner is not a
    geometric "bottom-left" corner.  SPECFEM, however, assigns semantic local
    edges 1..4 = bottom, right, top, left.  This function only fixes polygon
    orientation; the cyclic start corner is fixed later from the explicit
    physical curves.
    """
    connect = np.asarray(connect, dtype=np.int64)
    if dim != 2 or connect.ndim != 2 or connect.shape[1] != 4:
        return connect
    if coord is None:
        # Historical fallback. The 2.2 path always passes coordinates now.
        return connect[:, [0, 3, 2, 1]]

    out = connect.copy()
    area2 = _quad_signed_area2(out, coord)
    if np.any(np.abs(area2) <= 1e-14):
        bad = np.flatnonzero(np.abs(area2) <= 1e-14)[:10]
        raise ValueError(f"Degenerate Quad4 element(s) with zero signed area: {bad.tolist()}")

    flip = area2 < 0.0
    if np.any(flip):
        out[flip] = out[flip][:, [0, 3, 2, 1]]

    area2_after = _quad_signed_area2(out, coord)
    if np.any(area2_after <= 0.0):
        bad = np.flatnonzero(area2_after <= 0.0)[:10]
        raise ValueError(f"Quad4 orientation could not be made positive: {bad.tolist()}")
    return out


def _pair_code(a, b):
    """Encode an unordered pair of 1-based node ids in one uint64."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    if np.any(hi >= np.uint64(1 << 32)):
        raise ValueError("Node ids >= 2^32 are not supported by the edge encoder")
    return (lo << np.uint64(32)) | hi


def _extract_physical_curve_groups_2_2(physical_names, element_groups, nodes):
    """Read all explicit LIN_2 physical curves from a Gmsh 2.2 mesh."""
    by_name = {}
    node_index = nodes.index

    for physical_tag, physic in physical_names.items():
        if physic.dim != 1:
            continue
        if physical_tag not in element_groups.groups:
            raise ValueError(
                f"1-D physical group {physic.name!r} (tag {physical_tag}) "
                "has no elements in the Gmsh file"
            )
        line_block = element_groups.get_group(physical_tag)
        bad = line_block[line_block.element_type != 'LIN_2']
        if len(bad):
            raise ValueError(
                f"1-D physical group {physic.name!r} contains non-LIN_2 elements: "
                f"{sorted(set(bad.element_type))}"
            )

        line_connect = np.stack(line_block.node_index, axis=0)
        line_connect = node_index.get_indexer(line_connect.ravel()).reshape(
            line_connect.shape
        ) + 1
        if np.any(line_connect <= 0):
            raise ValueError(
                f"Unknown node tag while reading physical curve {physic.name!r}"
            )

        codes = _pair_code(line_connect[:, 0], line_connect[:, 1])
        by_name[physic.name] = {
            "tag": int(physical_tag),
            "connect": line_connect.astype(np.int64),
            "codes": codes,
        }

    required = ("xmin", "xmax", "zmin", "zmax")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(
            "Missing required Gmsh Physical Curve(s): " + ", ".join(missing)
        )
    return by_name


def _target_edge_adjacency(block_names, connects, offsets, target_codes, chunk_size=500_000):
    """Find Quad4 sides matching only the requested physical-curve edges.

    This avoids building a Python dictionary for every edge of a multi-million
    element mesh.  Matching is vectorized and chunked.
    """
    target_codes = np.unique(np.asarray(target_codes, dtype=np.uint64))
    adjacency = {}

    for iblock, (block_name, connect, offset) in enumerate(
        zip(block_names, connects, offsets)
    ):
        nelem = connect.shape[0]
        for first in range(0, nelem, chunk_size):
            last = min(first + chunk_size, nelem)
            q = connect[first:last]

            for side0 in range(4):
                a = q[:, side0]
                b = q[:, (side0 + 1) % 4]
                codes = _pair_code(a, b)

                pos = np.searchsorted(target_codes, codes)
                valid = pos < target_codes.size
                hit = np.zeros(codes.shape, dtype=bool)
                if np.any(valid):
                    hit[valid] = target_codes[pos[valid]] == codes[valid]

                local_hits = np.flatnonzero(hit)
                for j in local_hits:
                    local_elem = first + int(j)
                    global_elem = int(offset) + local_elem + 1
                    code = int(codes[j])
                    adjacency.setdefault(code, []).append(
                        (
                            iblock,
                            block_name,
                            local_elem,
                            global_elem,
                            side0 + 1,
                        )
                    )
    return adjacency


def _rotate_quads(connect, rotations):
    """Apply per-element cyclic left rotations, preserving CCW orientation."""
    out = connect.copy()
    for r in (1, 2, 3):
        idx = np.flatnonzero(rotations == r)
        if idx.size:
            out[idx] = np.roll(out[idx], -r, axis=1)
    return out


def orient_and_build_sidesets_2_2(
    physical_names, element_groups, nodes, block_names, raw_connects, coord
):
    """Canonicalize boundary/interface Quad4 local edges for SPECFEM.

    SPECFEM's 2-D local edge semantics are:
        1 = bottom, 2 = right, 3 = top, 4 = left.

    Gmsh's Quad4 connectivity is cyclic but its first node is not guaranteed to
    be the geometric bottom-left corner.  Merely reversing CW -> CCW therefore
    leaves side numbers geometrically arbitrary.

    Explicit Gmsh physical curves provide the missing information.  We use them
    to rotate every touched Quad4 so that:
        zmin  -> side 1
        xmax  -> side 2
        zmax  -> side 3
        xmin  -> side 4
        water/sediment interface:
            water (socle_1) -> side 1
            sediment        -> side 3

    Cyclic rotations do not change element orientation or geometry.
    """
    dim = 2

    connects = [
        orient_connectivity_for_specfem(c, dim, coord=coord)
        for c in raw_connects
    ]
    offsets = [0] + np.cumsum([c.shape[0] for c in connects[:-1]]).tolist()

    curve_groups = _extract_physical_curve_groups_2_2(
        physical_names, element_groups, nodes
    )

    required_outer = ("xmin", "xmax", "zmin", "zmax")
    ordered_names = list(required_outer)
    ordered_names += [
        name for name in curve_groups
        if name not in required_outer
    ]

    all_target_codes = np.concatenate(
        [curve_groups[name]["codes"] for name in ordered_names]
    )
    adjacency = _target_edge_adjacency(
        block_names, connects, offsets, all_target_codes
    )

    rotations = [
        np.full(c.shape[0], -1, dtype=np.int8)
        for c in connects
    ]

    # side sets are assembled in physical-curve order.  We keep the exact
    # element ids found before rotation; only the local side id changes.
    side_sets = []
    side_faces = []
    side_names = []

    outer_expected = {
        "zmin": 1,
        "xmax": 2,
        "zmax": 3,
        "xmin": 4,
    }

    def impose_rotation(match, desired_side, curve_name, code):
        iblock, block_name, local_elem, global_elem, old_side = match
        rotation = (int(old_side) - int(desired_side)) % 4
        previous = int(rotations[iblock][local_elem])
        if previous >= 0 and previous != rotation:
            raise ValueError(
                "Conflicting SPECFEM local-edge constraints for "
                f"element {global_elem} ({block_name}) on curve {curve_name!r}: "
                f"previous rotation={previous}, new rotation={rotation}, "
                f"old side={old_side}, desired side={desired_side}, edge code={code}"
            )
        rotations[iblock][local_elem] = rotation
        return global_elem, int(desired_side)

    for name in ordered_names:
        group = curve_groups[name]
        ss = []
        sf = []

        if "|" not in name:
            desired = outer_expected[name]
            for code_u64 in group["codes"]:
                code = int(code_u64)
                matches = adjacency.get(code, [])
                if len(matches) != 1:
                    raise ValueError(
                        f"Outer-boundary edge {code} in {name!r} has "
                        f"{len(matches)} adjacent Quad4 sides instead of 1: {matches}"
                    )
                elem, side = impose_rotation(
                    matches[0], desired, name, code
                )
                ss.append(elem)
                sf.append(side)
        else:
            block_a, block_b = name.split("|", 1)
            if block_a not in block_names or block_b not in block_names:
                raise ValueError(
                    f"Coupling side set {name!r} refers to unknown block(s). "
                    f"Known blocks: {block_names}"
                )

            # This mesh construction always names the water block first and the
            # top sediment material second.
            if block_a != "socle_1" or not block_b.startswith("sed_vs_"):
                raise ValueError(
                    f"Unsupported coupling naming {name!r}; expected "
                    "'socle_1|sed_vs_*'"
                )

            for code_u64 in group["codes"]:
                code = int(code_u64)
                matches = adjacency.get(code, [])
                a_matches = [m for m in matches if m[1] == block_a]
                b_matches = [m for m in matches if m[1] == block_b]
                if len(a_matches) != 1 or len(b_matches) != 1:
                    raise ValueError(
                        f"Interface edge {code} in {name!r} does not match "
                        f"exactly one side in each block: "
                        f"{block_a}={a_matches}, {block_b}={b_matches}, "
                        f"all={matches}"
                    )

                elem_a, side_a = impose_rotation(
                    a_matches[0], 1, name, code
                )  # bottom of water
                elem_b, side_b = impose_rotation(
                    b_matches[0], 3, name, code
                )  # top of sediment
                ss.extend((elem_a, elem_b))
                sf.extend((side_a, side_b))

        side_sets.append(np.asarray(ss, dtype=np.int64))
        side_faces.append(np.asarray(sf, dtype=np.int64))
        side_names.append(name)

    # Untouched elements need no cyclic reindexing.
    for r in rotations:
        r[r < 0] = 0

    connects = [
        _rotate_quads(c, r)
        for c, r in zip(connects, rotations)
    ]

    # Strong post-condition: every side-set entry must now select exactly the
    # original physical-curve segment using SPECFEM's semantic side number.
    side_nodes = ((0, 1), (1, 2), (2, 3), (3, 0))
    for name, group, ss, sf in zip(
        ordered_names,
        [curve_groups[n] for n in ordered_names],
        side_sets,
        side_faces,
    ):
        expected_codes = group["codes"]
        if "|" not in name:
            actual_codes = []
            for elem, side in zip(ss, sf):
                # Locate block from the global Exodus element number.
                e0 = int(elem) - 1
                iblock = int(np.searchsorted(
                    np.asarray(offsets[1:] + [sum(c.shape[0] for c in connects)]),
                    e0,
                    side="right",
                ))
                local = e0 - int(offsets[iblock])
                q = connects[iblock][local]
                a, b = side_nodes[int(side) - 1]
                actual_codes.append(int(_pair_code(q[a], q[b])))
            if not np.array_equal(
                np.sort(np.asarray(actual_codes, dtype=np.uint64)),
                np.sort(expected_codes),
            ):
                raise ValueError(
                    f"Post-validation failed for side set {name!r}: "
                    "selected Quad4 sides do not reproduce the physical curve"
                )
        else:
            # Interface has two element-side entries per physical segment.
            actual_codes = []
            for j in range(0, len(ss), 2):
                pair_codes = []
                for k in (j, j + 1):
                    e0 = int(ss[k]) - 1
                    iblock = int(np.searchsorted(
                        np.asarray(offsets[1:] + [sum(c.shape[0] for c in connects)]),
                        e0,
                        side="right",
                    ))
                    local = e0 - int(offsets[iblock])
                    q = connects[iblock][local]
                    a, b = side_nodes[int(sf[k]) - 1]
                    pair_codes.append(int(_pair_code(q[a], q[b])))
                if pair_codes[0] != pair_codes[1]:
                    raise ValueError(
                        f"Post-validation failed for interface {name!r}: "
                        f"paired sides select different edges {pair_codes}"
                    )
                actual_codes.append(pair_codes[0])
            if not np.array_equal(
                np.sort(np.asarray(actual_codes, dtype=np.uint64)),
                np.sort(expected_codes),
            ):
                raise ValueError(
                    f"Post-validation failed for interface {name!r}: "
                    "selected sides do not reproduce the physical curve"
                )

    # Cyclic rotations preserve positive orientation, nevertheless assert it.
    for block_name, connect in zip(block_names, connects):
        area2 = _quad_signed_area2(connect, coord)
        if np.any(area2 <= 0.0):
            bad = np.flatnonzero(area2 <= 0.0)[:10]
            raise ValueError(
                f"Non-positive Quad4 orientation after canonicalization in "
                f"{block_name!r}: {bad.tolist()}"
            )

    print("SPECFEM local-edge canonicalization:")
    print("  zmin -> side 1 (bottom)")
    print("  xmax -> side 2 (right)")
    print("  zmax -> side 3 (top)")
    print("  xmin -> side 4 (left)")
    print("  water/sediment -> water side 1, sediment side 3")

    return connects, side_sets, side_faces, side_names

def parse_2_2(file, binary: bool, size_of_size_t=8, endian="little"):
    out = dict()
    for line in file:
        line = (str(line, 'utf-8') if binary else line).rstrip()
        if not line.startswith('$'):
            raise ValueError(f"'{line}' does not starts with a $. It is not a section header")
        if line == "$PhysicalNames":
            out['PhysicalNames'] = parse_physical_names(file, binary, cache=out.get('PhysicalNames'))
        elif line == "$Elements":
            if binary:
                out['Elements'] = parse_elements_2_2_binary(
                    file, size_of_size_t, endian, cache=out.get('Elements')
                )
            else:
                out['Elements'] = parse_elements_2_2_ascii(file, cache=out.get('Elements'))
        elif line == "$Nodes":
            if binary:
                out['Nodes'] = parse_nodes_2_2_binary(
                    file, size_of_size_t, endian, cache=out.get('Nodes')
                )
            else:
                out['Nodes'] = parse_nodes_2_2_ascii(file, cache=out.get('Nodes'))
        elif line == "$ElementData":
            parse_element_data()
        elif line == "$ElementNodeData":
            parse_element_node_data()
        elif line == "$InterpolationScheme":
            parse_interpolation_scheme()
        else:
            raise ValueError(f"Unknown section header'{line}'")

    if 'PhysicalNames' not in out:
        raise ValueError("$PhysicalNames section was not found.")
    if 'Nodes' not in out:
        raise ValueError("$Nodes section was not found.")
    if 'Elements' not in out:
        raise ValueError("$Elements section was not found.")

    groups = out['Elements'].groupby('entity_tag')

    # 2-D physical groups become Exodus element blocks.
    names = []
    raw_connects = []
    for physical_tag, physic in out['PhysicalNames'].items():
        if physic.dim != 2:
            continue
        if physical_tag not in groups.groups:
            raise ValueError(
                f"2-D physical group {physic.name!r} (tag {physical_tag}) "
                "has no elements in the Gmsh file"
            )

        block = groups.get_group(physical_tag)
        bad = block[block.element_type != 'QUA_4']
        if len(bad):
            raise ValueError(
                f"2-D physical group {physic.name!r} contains non-Quad4 elements: "
                f"{sorted(set(bad.element_type))}"
            )

        connect = np.stack(block.node_index, axis=0)
        connect = out['Nodes'].index.get_indexer(connect.ravel()).reshape(connect.shape) + 1
        if np.any(connect <= 0):
            raise ValueError(f"Unknown node tag while reading block {physic.name!r}")

        names.append(physic.name)
        raw_connects.append(connect.astype(np.int64))

    if not raw_connects:
        raise ValueError("No 2-D physical element blocks were found in the Gmsh mesh")

    dim = 2
    coord = [out['Nodes'].x.values, out['Nodes'].y.values]

    # Canonicalize local Quad4 numbering from the explicit Gmsh physical
    # curves.  SPECFEM assigns semantic local edges
    # 1=bottom, 2=right, 3=top, 4=left, while Gmsh does not guarantee that
    # node 1 is the geometric bottom-left corner.
    connects, side_sets, side_faces, side_names = orient_and_build_sidesets_2_2(
        out['PhysicalNames'],
        groups,
        out['Nodes'],
        names,
        raw_connects,
        coord,
    )

    print("Physical 2-D blocks:", names)
    print("Side sets from explicit Gmsh physical curves:")
    for ss_name, ss in zip(side_names, side_sets):
        if ss_name.startswith("socle_1|"):
            detail = f"{len(ss)//2} interface edges / {len(ss)} element-sides"
        else:
            detail = f"{len(ss)} boundary edges"
        print(f"  {ss_name}: {detail}")

    return names, connects, coord, side_sets, side_faces, side_names


def parse_4_1(file, binary: bool, size_of_size_t=8, endian="little"):
    out = dict()
    for line in file:
        line = (str(line, 'utf-8') if binary else line).rstrip()
        if not line.startswith('$'):
            raise ValueError(f"'{line}' does not starts with a $. It is not a section header")
        if line == "$PhysicalNames":
            out['PhysicalNames'] = parse_physical_names(file, binary, cache=out.get('PhysicalNames'))
        elif line == "$Elements":
            if binary:
                out['Elements'] = parse_elements_4_1_binary(file, size_of_size_t, endian, cache=out.get('Elements'))
            else:
                out['Elements'] = parse_elements_4_1_ascii(file, cache=out.get('Elements'))
        elif line == "$Nodes":
            if binary:
                out['Nodes'] = parse_nodes_4_1_binary(file, size_of_size_t, endian, cache=out.get('Nodes'))
            else:
                out['Nodes'] = parse_nodes_4_1_ascii(file, cache=out.get('Nodes'))
        elif line == "$GhostElements":
            parse_ghost_elements()
        elif line == "$Parametrizations":
            parse_parametrizations()
        elif line == "$ElementData":
            parse_element_data()
        elif line == "$ElementNodeData":
            parse_element_node_data()
        elif line == "$InterpolationScheme":
            parse_interpolation_scheme()
        elif line == "$Entities":
            if binary:
                out['Entities'] = parse_entities_binary(file, size_of_size_t, endian, cache=out.get('Entities'))
            else:
                out['Entities'] = parse_entities_ascii(file, cache=out.get('Entities'))
        else:
            raise ValueError(f"Unknown section header'{line}'")

    if 'Nodes' not in out or 'Elements' not in out:
        raise ValueError("$Nodes or $Elements section was not found")

    names = []
    raw_connects = []
    groups = out['Elements'].groupby(['entity_dim', 'entity_tag'])
    for n, p in out['PhysicalNames'].items():
        if p.dim != 2:
            continue
        names.append(p.name)
        connect = np.stack(groups.get_group((p.dim, n)).node_index, axis=0)
        connect = out['Nodes'].index.get_indexer(connect.ravel()).reshape(connect.shape) + 1
        raw_connects.append(connect)

    dim = 2
    coord = [out['Nodes'].x.values, out['Nodes'].y.values]
    offsets = [0] + np.cumsum([c.shape[0] for c in raw_connects[:-1]]).tolist()
    border_sides, border_faces, border_names = make_borders(out['Nodes'], raw_connects, dim)
    interface_sets, interface_faces, interface_names = make_water_sediment_interfaces(
        names, raw_connects, offsets, dim
    )
    connects = [orient_connectivity_for_specfem(c, dim) for c in raw_connects]
    border_faces = [remap_sides_for_specfem(sf, dim) for sf in border_faces]
    interface_faces = [remap_sides_for_specfem(sf, dim) for sf in interface_faces]

    side_sets = list(border_sides) + interface_sets
    side_faces = list(border_faces) + interface_faces
    side_names = list(border_names) + interface_names

    print("Physical 2-D blocks:", names)
    print("Side sets:")
    for ss_name, ss in zip(side_names, side_sets):
        print(f"  {ss_name}: {len(ss)} entries")

    return names, connects, coord, side_sets, side_faces, side_names

def _quad_edge_table(connectivity, offset=0):
    """Return sorted Quad4 edge keys, global element ids and Exodus side ids."""
    connectivity = np.asarray(connectivity)
    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise ValueError("Quad4 connectivity with shape (nelem, 4) is required")

    local_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    edges = np.sort(connectivity[:, local_edges], axis=2).reshape(-1, 2)
    nelem = connectivity.shape[0]
    elems = np.repeat(np.arange(nelem, dtype=np.int64) + int(offset) + 1, 4)
    sides = np.tile(np.arange(1, 5, dtype=np.int64), nelem)
    return edges, elems, sides


def _edge_keys(edges):
    """View each two-node edge as one sortable/comparable key."""
    edges = np.ascontiguousarray(edges)
    dtype = np.dtype([('n0', edges.dtype), ('n1', edges.dtype)])
    return edges.view(dtype).reshape(-1)


def make_sideset(nodes1, offset1, nodes2, offset2, dim):
    """Return the exact shared element sides between two element blocks."""
    if dim != 2:
        # Preserve the historical 3-D path.
        ind1, inv1 = np.unique(nodes1, return_inverse=True)
        ind2, inv2 = np.unique(nodes2, return_inverse=True)
        _, int1, int2 = np.intersect1d(ind1, ind2, assume_unique=True, return_indices=True)
        mask1 = np.isin(inv1, int1).reshape(-1, nodes1.shape[1])
        mask2 = np.isin(inv2, int2).reshape(-1, nodes2.shape[1])
        ind1 = np.argwhere(mask1.sum(-1) >= 2**(dim-1)).ravel()
        ind2 = np.argwhere(mask2.sum(-1) >= 2**(dim-1)).ravel()
        mask1 = mask1[ind1]
        mask2 = mask2[ind2]
        sort1 = np.lexsort(np.sort(nodes1[ind1][mask1].reshape(-1, 1 << (dim-1))).T)
        sort2 = np.lexsort(np.sort(nodes2[ind2][mask2].reshape(-1, 1 << (dim-1))).T)
        sideset = np.stack([ind1[sort1]+offset1+1, ind2[sort2]+offset2+1], axis=1).ravel()
        face_mask = np.array([1<<1 | 1<<4 | 1<<5,
                              1<<1 | 1<<2 | 1<<5,
                              1<<2 | 1<<3 | 1<<5,
                              1<<3 | 1<<4 | 1<<5,
                              1<<1 | 1<<4 | 1<<6,
                              1<<1 | 1<<2 | 1<<6,
                              1<<2 | 1<<3 | 1<<6,
                              1<<3 | 1<<4 | 1<<6])
        face_mask = np.broadcast_to(face_mask, mask1.shape)
        faces = np.stack([
            np.bitwise_and.reduce(face_mask[mask1[sort1]].reshape(-1, 1 << (dim-1)), axis=-1),
            np.bitwise_and.reduce(face_mask[mask2[sort2]].reshape(-1, 1 << (dim-1)), axis=-1)
        ], axis=1).ravel()
        return sideset, np.log2(faces).astype(int)

    edges1, elems1, sides1 = _quad_edge_table(nodes1, offset1)
    edges2, elems2, sides2 = _quad_edge_table(nodes2, offset2)
    _, i1, i2 = np.intersect1d(
        _edge_keys(edges1), _edge_keys(edges2),
        assume_unique=False, return_indices=True
    )
    if i1.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ss = np.column_stack((elems1[i1], elems2[i2])).ravel()
    sf = np.column_stack((sides1[i1], sides2[i2])).ravel()
    return ss.astype(np.int64), sf.astype(np.int64)


def remap_sides_for_specfem(side_ids, dim):
    """Map Gmsh Quad4 side ids after [0,3,2,1] connectivity reversal."""
    side_ids = np.asarray(side_ids, dtype=np.int64)
    if dim == 2:
        return (5 - side_ids).astype(np.int64)
    return side_ids



def side_sets_from_physical_curves_2_2(
    physical_names, element_groups, nodes, block_names, connects, offsets
):
    """Convert explicit Gmsh 2.2 Physical Curves into Exodus side sets.

    Each LIN_2 segment is matched against exact Quad4 edges from the already
    SPECFEM-oriented connectivity.

    Outer boundaries (xmin/xmax/zmin/zmax) touch one Quad4.
    Coupling curves named ``blockA|blockB`` touch one Quad4 from each named
    block; both element-sides are written in A,B order.
    """
    edge_lookup = {}
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))

    for block_name, connect, offset in zip(block_names, connects, offsets):
        for local_elem, quad in enumerate(connect):
            global_elem = int(offset) + local_elem + 1
            for side_id, (a, b) in enumerate(local_edges, start=1):
                key = tuple(sorted((int(quad[a]), int(quad[b]))))
                edge_lookup.setdefault(key, []).append(
                    (block_name, global_elem, side_id)
                )

    curve_groups = [
        (tag, physic)
        for tag, physic in physical_names.items()
        if physic.dim == 1
    ]
    if not curve_groups:
        raise ValueError(
            "No 1-D Physical Curves were found in the Gmsh file. "
            "Regenerate Isfjorden.msh with the updated isfjorden_msh.py."
        )

    by_name = {physic.name: (tag, physic) for tag, physic in curve_groups}
    required_outer = ("xmin", "xmax", "zmin", "zmax")
    missing = [name for name in required_outer if name not in by_name]
    if missing:
        raise ValueError(
            "Missing required Gmsh Physical Curve(s): " + ", ".join(missing)
        )

    # Keep historical boundary ordering, then append the coupling curves.
    ordered = [by_name[name] for name in required_outer]
    ordered += [
        (tag, physic)
        for tag, physic in curve_groups
        if physic.name not in required_outer
    ]

    side_sets = []
    side_faces = []
    side_names = []
    node_index = nodes.index

    for physical_tag, physic in ordered:
        name = physic.name

        if physical_tag not in element_groups.groups:
            raise ValueError(
                f"1-D physical group {name!r} (tag {physical_tag}) "
                "has no elements in the Gmsh file"
            )

        line_block = element_groups.get_group(physical_tag)
        bad = line_block[line_block.element_type != 'LIN_2']
        if len(bad):
            raise ValueError(
                f"1-D physical group {name!r} contains non-LIN_2 elements: "
                f"{sorted(set(bad.element_type))}"
            )

        line_connect = np.stack(line_block.node_index, axis=0)
        line_connect = node_index.get_indexer(line_connect.ravel()).reshape(
            line_connect.shape
        ) + 1
        if np.any(line_connect <= 0):
            raise ValueError(f"Unknown node tag while reading side set {name!r}")

        ss = []
        sf = []
        is_interface = "|" in name

        if is_interface:
            block_a, block_b = name.split("|", 1)
            if block_a not in block_names or block_b not in block_names:
                raise ValueError(
                    f"Coupling side set {name!r} refers to unknown block(s). "
                    f"Known 2-D blocks: {block_names}"
                )

        for segment in line_connect:
            if len(segment) != 2:
                raise ValueError(
                    f"Physical Curve {name!r} contains an element with "
                    f"{len(segment)} nodes; LIN_2 was expected"
                )

            key = tuple(sorted((int(segment[0]), int(segment[1]))))
            matches = edge_lookup.get(key, [])

            if is_interface:
                a_matches = [m for m in matches if m[0] == block_a]
                b_matches = [m for m in matches if m[0] == block_b]

                if len(a_matches) != 1 or len(b_matches) != 1:
                    raise ValueError(
                        f"Interface segment {key} in {name!r} does not match "
                        f"exactly one side in each block: "
                        f"{block_a}={a_matches}, {block_b}={b_matches}, "
                        f"all matches={matches}"
                    )

                for _, elem, side in (a_matches[0], b_matches[0]):
                    ss.append(elem)
                    sf.append(side)
            else:
                if len(matches) != 1:
                    raise ValueError(
                        f"Outer-boundary segment {key} in {name!r} has "
                        f"{len(matches)} adjacent Quad4 sides instead of 1: "
                        f"{matches}"
                    )

                _, elem, side = matches[0]
                ss.append(elem)
                sf.append(side)

        if not ss:
            raise ValueError(f"Physical Curve {name!r} produced an empty side set")

        side_sets.append(np.asarray(ss, dtype=np.int64))
        side_faces.append(np.asarray(sf, dtype=np.int64))
        side_names.append(name)

    return side_sets, side_faces, side_names

def make_water_sediment_interfaces(names, connects, offsets, dim):
    """Build only the water/top-sediment coupling interfaces.

    The sediment is split into several physical material blocks (sed_vs_*).
    We therefore create one side set per sediment material that actually
    touches water, e.g. ``socle_1|sed_vs_180``.  Contacts between sediment
    bins, and the sediment/basement contact, are deliberately ignored.
    """
    if dim != 2:
        raise NotImplementedError("Water/sediment coupling helper currently supports 2-D meshes only")

    normalized_names = [str(name).strip().strip('\"').strip("'") for name in names]
    try:
        water_idx = normalized_names.index('socle_1')
    except ValueError as exc:
        raise ValueError(
            "Physical block 'socle_1' (water) was not found. "
            f"Available blocks: {normalized_names!r}"
        ) from exc

    sediment_idx = [i for i, name in enumerate(normalized_names) if name.startswith('sed_vs_')]
    if not sediment_idx:
        raise ValueError(
            "No sediment physical block named 'sed_vs_*' was found. "
            f"Available blocks: {normalized_names!r}"
        )

    side_sets = []
    side_faces = []
    side_names = []
    total_shared_edges = 0

    for i in sediment_idx:
        ss, sf = make_sideset(
            connects[water_idx], offsets[water_idx],
            connects[i], offsets[i],
            dim
        )
        if len(ss) == 0:
            continue
        side_sets.append(ss)
        side_faces.append(sf)
        side_names.append(f"socle_1|{normalized_names[i]}")
        total_shared_edges += len(ss) // 2

    if not side_sets:
        raise ValueError(
            "No conforming water/sediment Quad4 edge was found. "
            "The water bottom and sediment top must share the same Gmsh edges/nodes."
        )

    print(
        f"Water/sediment coupling: {total_shared_edges} shared edges "
        f"across {len(side_sets)} sediment material side set(s)"
    )
    return side_sets, side_faces, side_names

def make_border(nodes, connect_all, face_mask, dim):
    mask = np.isin(connect_all-1, nodes)
    sideset, = np.where(mask.sum(-1) >= 2**(dim - 1))
    mask = mask[sideset]
    faces = np.bitwise_and.reduce(np.broadcast_to(face_mask, mask.shape)[mask].reshape(-1, 1 << (dim - 1)), axis=-1)
    return sideset+1, np.log2(faces).astype(int)  # +1 for exodus indexing starting at 1


def make_borders(coords, connects, dim):
    connect_all = np.concatenate(connects, axis=0)
    if dim == 2:
        names = ['xmin', 'xmax', 'zmin', 'zmax']
    else:
        names = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']

    if dim == 2:
        face_mask = np.array([1<<1 | 1<<4,
                              1<<1 | 1<<2,
                              1<<2 | 1<<3,
                              1<<3 | 1<<4])  # bit mask for each Quad corner to edges it's touching
    else:
        face_mask = np.array([1<<1 | 1<<4 | 1<<5,
                              1<<1 | 1<<2 | 1<<5,
                              1<<2 | 1<<3 | 1<<5,
                              1<<3 | 1<<4 | 1<<5,
                              1<<1 | 1<<4 | 1<<6,
                              1<<1 | 1<<2 | 1<<6,
                              1<<2 | 1<<3 | 1<<6,
                              1<<3 | 1<<4 | 1<<6])  # bit mask for each Hex corner to faces it's touching
    sidesets = list()
    faces = list()

    sideset, face = make_border(np.where(coords.x == coords.x.min())[0], connect_all, face_mask, dim)
    sidesets.append(sideset)
    faces.append(face)

    sideset, face = make_border(np.where(coords.x == coords.x.max())[0], connect_all, face_mask, dim)
    sidesets.append(sideset)
    faces.append(face)

    sideset, face = make_border(np.where(coords.y == coords.y.min())[0], connect_all, face_mask, dim)
    sidesets.append(sideset)
    faces.append(face)

    sideset, face = make_border(np.where(coords.y == coords.y.max())[0], connect_all, face_mask, dim)
    sidesets.append(sideset)
    faces.append(face)

    if dim == 3:
        sideset, face = make_border(np.where(coords.z == coords.z.min())[0], connect_all, face_mask, dim)
        sidesets.append(sideset)
        faces.append(face)

        sideset, face = make_border(np.where(coords.z == coords.z.max())[0], connect_all, face_mask, dim)
        sidesets.append(sideset)
        faces.append(face)

    return sidesets, faces, names


def add_string_attrs(attrs, name, value, length=None):
    value += "\0"
    tid = h5py.h5t.C_S1.copy()
    if length is None:
        tid.set_size(len(value))
    else:
        tid.set_size(length)
    attrs.create(name, value, dtype=h5py.Datatype(tid))


def add_dimension_scale_attrs(attrs, value):
    add_string_attrs(attrs, 'NAME', f"This is a netCDF dimension but not a netCDF variable. {value}", 64)
    attrs.create('_Netcdf4Dimid', add_dimension_scale_attrs.id_, dtype=np.int32)
    add_dimension_scale_attrs.id_ += 1
    # attrs.create('REFENCE_LIST')


add_dimension_scale_attrs.id_ = 0


def add_dimension_scale(store, name, value):
    dataset = store.create_dataset(name, dtype=">f4", shape=value, fillvalue=0)
    dataset.make_scale()
    add_dimension_scale_attrs(dataset.attrs, value)


def add_dimension_ref(store, dataset, dimension: list):
    dataset.attrs.create("_Netcdf4Coordinates", [get_id(store, n) for n in dimension], dtype=np.int32)
    for dim, name in enumerate(dimension):
        dataset.dims[dim].attach_scale(store[name])


def get_id(dataset, name):
    return dataset[name].attrs["_Netcdf4Dimid"]




def export_exodus(path, full_exodus, names, connects, coord, side_sets, side_faces, side_names):
    out = h5py.File(path, 'w', track_order=True)
    total_elements = sum(c.__len__() for c in connects)

    if full_exodus:
        out.attrs['_nc3_strict'] = 1
        out.attrs['pi_version'] = [8.029999732971191]
        out.attrs['version'] = [8.029999732971191]
        out.attrs['floating_point_word_size'] = [8]
        out.attrs['file_size'] = [1]
        out.attrs['maximum_name_length'] = [32]
        out.attrs['int64_status'] = [0]
        add_string_attrs(out.attrs, '_NCProperties', "version=2,netcdf=4.7.4,hdf5=1.12.1")
        add_dimension_scale(out, 'len_name', 256)
        add_dimension_scale(out, 'time_step', 0)
        add_dimension_scale(out, 'num_dim', len(coord))
        add_dimension_scale(out, 'num_nodes', coord[0].size)
        add_dimension_scale(out, 'num_elem', total_elements)
        add_dimension_scale(out, 'num_el_blk', len(connects))
        add_dimension_scale(out, 'num_side_sets', len(side_sets))

        out.create_dataset('time_whole', dtype=">f4", shape=0)

        out.create_dataset('eb_status', dtype="<i4", data=np.full(len(connects), 1))
        out.create_dataset('eb_prop1', dtype="<i4", data=np.arange(len(connects))+1)
        add_string_attrs(out['eb_prop1'].attrs, 'name', 'ID')

        out.create_dataset('ss_status', dtype="<i4", data=np.full(len(side_sets), 1))
        out.create_dataset('ss_prop1', dtype="<i4", data=np.arange(len(side_sets))+1)
        add_string_attrs(out['ss_prop1'].attrs, 'name', 'ID')

    out.create_dataset('coordx', dtype="<f8", data=coord[0])
    out.create_dataset('coordy', dtype="<f8", data=coord[1])
    if len(coord) == 3:
        out.create_dataset('coordz', dtype="<f8", data=coord[2])

    name_dst = out.create_dataset('eb_names', dtype="|S1", data=np.array([list(n.ljust(256, '\0')) for n in names], dtype="|S1"))
    if full_exodus:
        add_dimension_ref(out, name_dst, ['num_el_blk', 'len_name'])
        add_string_attrs(name_dst.attrs, "_FillValue", "")

    side_dst = out.create_dataset('ss_names', dtype="|S1", data=np.array([list(n.ljust(256, '\0')) for n in side_names], dtype="|S1"))
    if full_exodus:
        add_dimension_ref(out, side_dst, ['num_side_sets', 'len_name'])
        add_string_attrs(side_dst.attrs, "_FillValue", "")

    if len(coord) == 2:
        coor_dst = out.create_dataset('coor_names', dtype="|S1", data=np.array([list(a.ljust(256, '\0')) for a in 'xy'], dtype="|S1"))
    else:
        coor_dst = out.create_dataset('coor_names', dtype="|S1", data=np.array([list(a.ljust(256, '\0')) for a in 'xyz'], dtype="|S1"))
    if full_exodus:
        add_dimension_ref(out, coor_dst, ['num_dim', 'len_name'])
        add_string_attrs(coor_dst.attrs, "_FillValue", "")

    if full_exodus:
        for i, c in enumerate(connects, start=1):
            add_dimension_scale(out, f'num_el_in_blk{i}', len(c))
            add_dimension_scale(out, f'num_nod_per_el{i}', c.shape[1])

        if len(coord) == 2:
            if connects[0].shape[1] == 4:  # Quad4
                side_size = 2
                element_type = 'QUAD4'
            elif connects[0].shape[1] == 8:  # Quad8
                side_size = 3
                element_type = 'QUAD8'
            elif connects[0].shape[1] == 9:  # Quad9
                side_size = 3
                element_type = 'QUAD9'
            else:
                print(f"Unknown element type with {connects[0].shape[1]} in 2D.")
                return 2
        else:
            if connects[0].shape[1] == 8:  # Hex8
                side_size = 4
                element_type = 'HEX8'
            elif connects[0].shape[1] == 20:  # Hex20
                side_size = 8
                element_type = 'HEX20'
            elif connects[0].shape[1] == 27:  # Hex27
                side_size = 9
                element_type = 'HEX27'
            elif connects[0].shape[1] == 64:  # Hex64
                print(f"Hex64 is not supported.")
                return 2
            elif connects[0].shape[1] == 125:  # Hex125
                print(f"Hex125 is not supported.")
                return 2
            else:
                print(f"Unknown element type with {connects[0].shape[1]} in 3D.")
                return 2

    for i, connect in enumerate(connects, start=1):
        connect_dst = out.create_dataset(f'connect{i}', dtype="<i4", data=connect)
        if full_exodus:
            add_dimension_ref(out, connect_dst, [f'num_el_in_blk{i}', f'num_nod_per_el{i}'])
            add_string_attrs(connect_dst.attrs, "elem_type", element_type)
    for i, (ss, sf) in enumerate(zip(side_sets, side_faces), start=1):
        if full_exodus:
            add_dimension_scale(out, f'num_side_ss{i}', len(ss))
            add_dimension_scale(out,f'num_df_ss{i}', side_size*len(ss))
        elem_ss_dst = out.create_dataset(f'elem_ss{i}', dtype="<i4", data=ss)
        side_ss_dst = out.create_dataset(f'side_ss{i}', dtype="<i4", data=sf)
        if full_exodus:
            add_dimension_ref(out, elem_ss_dst, [f'num_side_ss{i}'])
            add_dimension_ref(out, side_ss_dst, [f'num_side_ss{i}'])
            dist_dst = out.create_dataset(f'dist_fac_ss{i}', dtype="<f8", data=np.full(side_size*len(ss), 1.0))
            add_dimension_ref(out, dist_dst, [f'num_df_ss{i}'])


    if full_exodus:
        add_dimension_scale(out, 'num_qa_rec', 1)
        add_dimension_scale(out, 'four', 4)

        qa_len = 33
        add_dimension_scale(out, 'len_string', qa_len)

        import time
        t = time.localtime()
        date = time.strftime('%d/%m/%Y', t)
        hour = time.strftime('%H:%M:%S', t)
        qa_dst = out.create_dataset('qa_records', dtype="|S1",
                                    data=np.array([[list('CUBIT'.ljust(qa_len, '\0')),
                                                    list('2023.11'.ljust(qa_len, '\0')),
                                                    list(date.ljust(qa_len, '\0')),
                                                    list(hour.ljust(qa_len, '\0'))]], dtype="|S1"))
        add_string_attrs(out.attrs, 'title', f"cubit({path.split('/')[-1]}): {date}: {hour}")
        add_dimension_ref(out, qa_dst, ['num_qa_rec', 'four', 'len_string'])

        elem_dst = out.create_dataset('elem_map', dtype="<i4", data=np.arange(total_elements)+1)
        add_dimension_ref(out, elem_dst, ['num_elem'])
        elem_map_dst = out.create_dataset('elem_num_map', dtype="<i4", data=np.arange(total_elements)+1)
        add_dimension_ref(out, elem_map_dst, ['num_elem'])
        node_dst = out.create_dataset('node_num_map', dtype="<i4", data=np.arange(coord[0].size)+1)
        add_dimension_ref(out, node_dst, ['num_nodes'])
    return 0


def main(args):
    if args.output is None:
        args.output = args.input.rsplit('.', 1)[0] + ".e"
    if (not args.erase) and os.path.isfile(args.output):
        print("Output file already exist and erase flag is not set:", args.output, file=sys.stderr)
        return 1
    with open(args.input, 'rb') as f:
        if f.readline().rstrip() != b'$MeshFormat':
            print("Input file does not start with '$MeshFormat'.", file=sys.stderr)
            return 2
        version, binary, size_of_size_t = f.readline().rstrip().split()

        if binary := binary == b'1':
            one = f.readline().rstrip()
            if int.from_bytes(one, byteorder='little', signed=False) == 1:
                endian = 'little'
            elif int.from_bytes(one, byteorder='big', signed=False) == 1:
                endian = 'big'
            else:
                print("Unsupported endian", file=sys.stderr)
                return 2
        size_of_size_t = int(size_of_size_t)

        if f.readline().rstrip() != b'$EndMeshFormat':
            print("Input file '$MeshFormat' section does not end with '$EndMeshFormat'.", file=sys.stderr)
            return 2
        if version == b'2.2':
            if binary:
                parse_file = parse_2_2(f, binary, size_of_size_t, endian)
            else:
                f.close()
                with open(args.input, 'r') as f2:
                    f2.readline()
                    f2.readline()
                    f2.readline()
                    parse_file = parse_2_2(f2, binary)
        elif version == b'4.1':
            if binary:
                parse_file = parse_4_1(f, binary, size_of_size_t, endian)
            else:
                f.close()
                with open(args.input, 'r') as f2:
                    f2.readline()
                    f2.readline()
                    f2.readline()
                    parse_file = parse_4_1(f2, binary)
        else:
            print("Unknown msh file version:", version, file=sys.stderr)

    # Always write a complete Exodus-II file.
    #
    # The reduced/minimal output omits the `elem_type` attribute on connect*
    # datasets and omits explicit block/set metadata. With many element blocks
    # this makes side-set interpretation ambiguous for Exodus readers.
    # Side sets therefore require full Exodus-II metadata.
    if not args.full:
        print("Writing full Exodus-II metadata (required for reliable side sets).")
    return export_exodus(args.output, True, *parse_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")
    parser.add_argument("-e", "--erase", action='store_true', help="Erase output file if it already exists")
    parser.add_argument("-f", "--full", action="store_true", help="Deprecated: full Exodus-II metadata is now always written")

    sys.exit(main(parser.parse_args()))
