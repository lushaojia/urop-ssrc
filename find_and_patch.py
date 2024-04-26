import bpy
import bmesh
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--inpath',
        type=str)
    parser.add_argument(
        '-o',
        '--outpath',
        type=str)
    return parser.parse_args()

def read_dotobj(file_path):
    model = {'v': [], 'vt': [], 'vn': [], 'f': []}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            if not line:
                continue

            if line[0] == 'v':
                model['v'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'vt':
                model['vt'].append((float(line[1]), float(line[2])))
            elif line[0] == 'vn':
                model['vn'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'f':
                v1 = tuple(int(idx) if idx else None for idx in line[1].split('/'))
                v2 = tuple(int(idx) if idx else None for idx in line[2].split('/') )
                v3 = tuple(int(idx) if idx else None for idx in line[3].split('/'))
                model['f'].append((v1, v2, v3))
    return model

def vertex_pairs(face):
    return [(face[0], face[1]),
            (face[0], face[2]),
            (face[1], face[2])]
            
def another_face(face, faces):
    faces = [other for other in faces if other != face]
    pairs = vertex_pairs(face)
    suspects = set()

    # If the pair does not exist in another face, we add them to the suspect set
    for pair in pairs:
        another_face = False
        for other in faces:
            if pair[0] in other and pair[1] in other:
                another_face = True
                break
        if not another_face:
            suspects.add(pair[0])
            suspects.add(pair[1])
    return suspects

def bound_holes(filepath):
    dot_obj = read_dotobj(filepath)
    faces = [tuple(vertext_info[0] for vertext_info in vertices) for vertices in dot_obj['f']]
    suspect_indices = set()
    for face in faces:
        for idx in another_face(face, faces):
            suspect_indices.add(idx)
    return suspect_indices


# find indices of vertices on edge of holes
#args = get_arguments() # path passed in as cmd line arg
#inpath = args.inpath
inpath = '/Users/lusha/Documents/School/MIT/UROP/SSRC/CHHA0001/cleaned_CHHA000102/600_vertices.obj' # absolute path
suspect_indices = bound_holes(inpath)

# get model
obj = bpy.data.objects[0]

# copy model
copy = obj.copy()
copy.data = obj.data.copy()
bpy.context.collection.objects.link(copy) # link it to scene

# set copy as active object
bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]


# get mesh
me = copy.data

# go into edit mode
bpy.ops.object.mode_set(mode='EDIT')

# get a fresh bmesh from this mesh
bm = bmesh.from_edit_mesh(me)
# get all vertices
vertices = [v for v in bm.verts]

# select vertices on edge of holes
for vertex in vertices:
    if vertex.index in suspect_indices:
        vertex.select = True
    else:
        vertex.select = False

bmesh.update_edit_mesh(me)

# Set the merge distance threshold
merge_distance = 0.01  # Adjust this value as needed

# Perform the Merge by Distance operation
bpy.ops.mesh.remove_doubles(threshold = merge_distance)

