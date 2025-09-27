import math
import random
import pandas as pd
import numpy as np
import alphashape
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.neighbors import KDTree
from .util import read_dotobj
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from tqdm import tqdm
from shapely.ops import unary_union



def run_pca_on_3D_vertices(dot_obj: dict):
    '''
    Run Principal Component Analysis (PCA) on the vertices of a 3D model specified by `dot_obj`.
    See description of the return value of `read_dotobj` for the format of `dot_obj`.

    Returns:
        Fitted PCA object and all principal components.
    '''
    vtcs = dot_obj['v']
    vtcs_df = pd.DataFrame(vtcs, columns=['x', 'y', 'z'])
    pca = PCA()
    pca.fit(vtcs_df)
    print(f"Percentage of variation explained: {pca.explained_variance_ratio_}")
    pcs_all = pca.transform(vtcs_df)
    # print(f"Dimensions:\n{vertices_pca.shape}")

    return pca, pcs_all
    
def flatten_3D_vertices_to_2pcs(pcs_all):
    '''
    Return the first two of all principal components.
    '''
    pts_2d = np.array([coord[:2] for coord in pcs_all])
    # print(f"Dimensions: {pts_2d.shape}")

    # # Visualize vertices mapped to 2D space
    # x, y = pts_2d.T
    # plt.scatter(x, -y, s=0.1)
    # plt.title("Garment Flattened (3D Vertices Mapped to 2D Space with PCA)")
    # plt.show()

    return pts_2d # np.ndarray


def extract_polygons(alpha_shape: alphashape.alphashape):
    '''
    Theoretically, alpha shape is supposed to be a convex hull (single polygon)
    of a given sest of points. However, in practice, depending on the alpha parameter,
    it may be a polygon with holes or multiple polygons.

    Returns:
        List of polygons in the alpha shape.
    '''
    # Handle all possible geometry types
    if isinstance(alpha_shape, Polygon):
        polygons = [alpha_shape]
    elif isinstance(alpha_shape, MultiPolygon):
        polygons = list(alpha_shape.geoms)
    elif isinstance(alpha_shape, GeometryCollection):
        for g in alpha_shape.geoms:
            if hasattr(g, "exterior"):
                x, y = g.exterior.xy
        polygons = [g for g in alpha_shape.geoms if isinstance(g, Polygon)]
    else:
        polygons = []
    # print(f"Number of polygons: {len(polygons)}")
    return polygons


def bound_garment(pts_2d: np.ndarray, alpha = 200) -> set[tuple[float, float]]:
    alpha_shape = alphashape.alphashape(pts_2d, alpha=alpha)
   
    polygons = extract_polygons(alpha_shape)

    # for polygon in polygons:
    #     # visualize the silhouette (i.e. alpha shape)
    #     plt.plot(*polygon.exterior.xy)
    # plt.title("Silhouette (i.e. Alpha Shape) of the Garment")
    # plt.gca().invert_yaxis()
    # plt.show()

    alpha_shape_coords = set()
    for polygon in polygons:
        x, y = polygon.exterior.coords.xy
        alpha_shape_coords.add(zip(x, y))
    return alpha_shape_coords


def make_KDtree(pts_2d):
    return KDTree(pts_2d, leaf_size=1)


def find_angle(origin, axis_pt, other_pt):
    vec1 = np.subtract(axis_pt, origin)
    vec2 = np.subtract(other_pt, origin)

    dot = np.dot(vec1, vec2)
    cross_z = vec1[0]*vec2[1] - vec1[1]*vec2[0]  # 2D cross product z-component
    angle = np.degrees(np.arctan2(cross_z, dot))  # range (-180, 180]
    return angle % 360  # convert to [0, 360)

    # vec1 = np.subtract(axis_pt, origin) # Treat as arbitrary x-axis
    # vec2 = np.subtract(other_pt, origin)

    # dot_product = np.dot(vec1, vec2)
    # v1_norm = np.linalg.norm(vec1)
    # v2_norm = np.linalg.norm(vec2)
    # # print(f"origin_pt: {origin}, axis_pt: {axis_pt}, other_pt: {other_pt}")
    # # print("Input to arccos", dot_product/(v1_norm * v2_norm))
    
    # angle = np.arccos(np.clip(dot_product/(v1_norm * v2_norm), -1, 1)) # np.arccos requires input ∈ [-1, 1]
    # angle = angle * (180/np.pi) # Radians to degrees, range (0, 180)

    # # Turn the 0–180° result from arccos into a 0–360° angle
    # # by determining if the rotation from vec1 to vec2 is clockwise or counterclockwise
    # if (vec1[1] * vec2[1]) < 0:
    #     angle = angle + 180
    # return angle % 360 # Get angle in [0, 360)

def max_consecutive_zeroes(arr):
    streak = False
    counts = [0]
    count = 0

    for num in arr:
        if streak:
            if num == 0:
                count += 1
            else:
                counts.append(count)
                count = 0
                streak = False
        else:
            if num == 0:
                count += 1
                streak = True
    return max(counts)


def is_surrounded360(nn_indices, pts_2d, th=24):
    # print(f"nn_indices: {nn_indices}")
    hash = [0] * 36 # 36 slots for 10 degree sectors
 
    origin_pt = pts_2d[nn_indices[0]]
    axis_pt = pts_2d[nn_indices[1]]

    other_pts = [pts_2d[idx] for idx in nn_indices[2:]]
    for pt in other_pts:
        angle = find_angle(origin_pt, axis_pt, pt)
        hash[math.floor(angle/10)] = 1
    # print("hash:", hash)
    return max_consecutive_zeroes(hash) < th

def mean_neighbor_dist(pts_2d, kdtree, num_ind=100, k=40):
    random_ind = random.sample(range(len(pts_2d)), num_ind)
    random_pts = pts_2d[random_ind, :]
    dist, ind = kdtree.query(random_pts, k=k)
    individual_mean = np.mean(dist, axis=1) # 1d array
    return np.mean(individual_mean) # scalar

def bound_holes(pts_2d):
    alpha_shape = bound_garment(pts_2d)
    kdtree = make_KDtree(pts_2d)
    mean_n2n_dist = mean_neighbor_dist(pts_2d, kdtree)
    bound_ind = []

    for i, pt in tqdm(enumerate(pts_2d)):
        if tuple(pt) in alpha_shape:
            continue 
        nn_ind = kdtree.query_radius(pt.reshape(1, 2), r=mean_n2n_dist)[0] # [array([idx1, ... , idxk])]
        if not is_surrounded360(nn_ind, pts_2d):
            bound_ind.append(i)
    # print("Number of points bounding holes:", len(bound_ind))
    return bound_ind

    

def debug():

    find_angle([0.03642289, 0.01385439],
               [0.03673349, 0.01392236],
               [0.03888722, 0.01763664])
    find_angle(np.array([0.03642289, 0.01385439]), 
               np.array([0.03673349, 0.01392236]), 
               np.array([0.04053209, 0.01475373]))


def pc_indices_to_3D_vertices(pca, vertices: list[tuple[float, float, float]], bound_indices: list[int]):
    """
    Map indices of PCA-transformed 2D points back to 3D vertices in original space
    
    Args:
        vertices: 3D vertices in original space (e.g. [(x1, y1, z1), (x2, y2, z2), ...])
        bound_indices: List of indices of bounding vertices in 2D space
    
    Returns:
        3D coordinates of boundary points
    """
    # Get the 3D PCA coordinates for boundary points
    bound_3d_pcs = vertices[bound_indices]
    
    # Transform back to original 3D space
    bound_3d_original = pca.inverse_transform(bound_3d_pcs)
    
    return bound_3d_original


def get_indices_of_vertices(dot_obj: dict, vertices: list[tuple[float, float, float]], tolerance=1e-6):
    '''
    Given a list of (x, y, z) vertices, return their indices in dot_obj['v'].
    '''
    indices = []
    for i, original_vertex in enumerate(dot_obj['v']):
        for vertex in vertices:
            if all(abs(a - b) < tolerance for a, b in zip(original_vertex, vertex)):
                indices.append(i)
                break
    return indices


def bound_holes_pca_based(obj_filepath):
    dot_obj = read_dotobj(obj_filepath)
    pca, pcs_all = run_pca_on_3D_vertices(dot_obj)
    pcs_2d = flatten_3D_vertices_to_2pcs(pcs_all)
    bound_indices = bound_holes(pcs_2d)
    bound_3d_vertices = pc_indices_to_3D_vertices(pca, pcs_all, bound_indices)
    indices = get_indices_of_vertices(dot_obj, bound_3d_vertices)

    return indices


if __name__ == "__main__":
    obj_filepath = './garment_models/6000_vertices.obj'
    bound_holes_pca_based(obj_filepath)