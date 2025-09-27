from .util import read_dotobj


def vertex_pairs(face: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> list:
    """
    Args:
        face: a three-element tuple of (index, texture, normal) for each vertex in the triangular face
    Returns:
        All possible pairs of vertices in this face
    """
    vtx1 = face[0]
    vtx2 = face[1]
    vtx3 = face[2]

    return [(vtx1[0], vtx2[0]),
            (vtx1[0], vtx3[0]),
            (vtx2[0], vtx3[0])]

def faces_to_vertices(dot_obj: dict):
    """
    Maps each vertex to the set of faces that it belongs to.
    Vertices and faces are identified by indices.

    Returns:
    {
        vertex_index_1: {face_index_11, face_index_12, ...},
        vertex_index_2: {face_index_21, face_index_22,...},
        ...
    } where `dot_obj['v'][vertex_index_i]` is a vertex for i = 0, 1, ...
    and the vertex with index vertex_index_i is on the edge of faces with indices face_index_ij
    """
    map = dict()
    for i, faces in enumerate(dot_obj['f']):
        num_vertices_on_face = 3
        for v in range(num_vertices_on_face):
            vertex_index = faces[v][0]
            if vertex_index in map:
                map[vertex_index].add(i)
            else:
                map[vertex_index] = {i}
    
    return map


def on_shared_edge(v1: int, v2: int, f: int, vertices_to_faces: dict[int, set[int]]) -> bool:
    """
    Given a pair of vertices on a face, determine whether the edge they form is on another another face.

    Args:
        v1: a vertex identified by index
        v2: another vertex identified by index
        f: index of a face that has an edge formed by v1 and v2
    """
    set1 = vertices_to_faces[v1]
    set2 = vertices_to_faces[v2]
    assert f in set1, f"expected vertex {v1} to be on face {f}"
    assert f in set2, f"expected vertex {v2} to be on face {f}"
    return len((set1 & set2) - {f}) == 1

def bound_holes_edge_based(file_path) -> set[int]:
    """
    Find the set of vertices that are suspected to bound simple holes 
    in the 3D geometry specified by the OBJ file at file_path.
    
    Simple holes are open gaps in the mesh where an edge is only on one face (instead of two).

    args:
        file_path (str): path to OBJ file
    """
    dot_obj = read_dotobj(file_path)
    map = faces_to_vertices(dot_obj)
    boundary_vertices = set()

    for i, face in enumerate(dot_obj['f']):
        pairs = vertex_pairs(face)
      
        for pair in pairs:
            v1 = pair[0]
            v2 = pair[1]
            if on_shared_edge(v1, v2, i, map):
                break
            else:
                boundary_vertices.add(v1)
                boundary_vertices.add(v2)
    
    return boundary_vertices


if __name__ == "__main__":
    obj_filepath = './garment_models/600_vertices.obj'
    boundary_vertices = bound_holes_edge_based(obj_filepath)
    for vertex in boundary_vertices:
        print(vertex)