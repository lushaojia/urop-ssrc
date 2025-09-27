def read_dotobj(file_path):
    """
    Args:
        file_path (str): path to OBJ file (https://www.cs.cmu.edu/~mbz/personal/graphics/obj.html)
    Requires:
        * The OBJ file to contain data from a 3D geometry wtih triangular faces
    Returns:
        A dictionary containing data read from the OBJ file where
            * dot_obj['v'][i] = (x, y, z) is the ith vertex (x, y, z) listed in the file

            * dot_obj['vt'][i] = (U, V) are the horizontal axis (U) 
            and vertical axis (V) mappings for the ith vertex

            * dot_obj['vn'][i] = (x, y, z) are the x, y, z components of 
            the normal vector for the ith vertex

            * dot_obj['f'][i][v] is a three-element tuple (index, texture, normal) for a vertex
            on the edge of the ith face for v = 0, 1, 2.
            Note that in this internal representation, vertices are zero-indexed 
            (but in the OBJ file, they come one-indexed).
    """
    dot_obj = {'v': [], 'vt': [], 'vn': [], 'f': []}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            if not line:
                continue

            if line[0] == 'v':
                dot_obj['v'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'vt':
                dot_obj['vt'].append((float(line[1]), float(line[2])))
            elif line[0] == 'vn':
                dot_obj['vn'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'f':
                v1 = tuple(int(idx) - 1 if idx else None for idx in line[1].split('/')) # OBJ uses one-indexing, but use zero-indexing internally
                v2 = tuple(int(idx) - 1 if idx else None for idx in line[2].split('/') )
                v3 = tuple(int(idx) - 1 if idx else None for idx in line[3].split('/'))
                dot_obj['f'].append((v1, v2, v3))
    return dot_obj