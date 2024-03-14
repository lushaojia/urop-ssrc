import math
import random
import pickle
import pandas as pd
import numpy as np
import alphashape
from matplotlib import pyplot as plt
from descartes import PolygonPatch
from sklearn.decomposition import PCA 
from sklearn.neighbors import KDTree
from util import read_dotobj, get_arguments

def original2pcs(args):
    vtcs = read_dotobj(args.obj_path)['v']
    vtcs_df = pd.DataFrame(vtcs, columns=['x', 'y', 'z'])
    pca = PCA()
    pca.fit(vtcs_df)
    print(f"Percentage of variation explained: {pca.explained_variance_ratio_}")
    pcs_all = pca.transform(vtcs_df)
    # print(f"Dimensions:\n{vertices_pca.shape}")
    pts_2d = np.array([coord[:2] for coord in pcs_all])
    print(f"Dimensions: {pts_2d.shape}")

    # Visualize vertices mapped to 2D space
    x, y = pts_2d.T
    plt.scatter(x,y, s=0.1)
    plt.show()

    return pcs_all, pts_2d # np.ndarray

def bound_garment(pts_2d, alpha=200):
    alpha_shape = alphashape.alphashape(pts_2d, alpha=alpha)
    # Visualize the silhouette (i.e. alpha shape)
    plt.plot(*alpha_shape.exterior.xy)
    plt.show()

    # Get points in alpha shape
    x, y = alpha_shape.exterior.coords.xy
    return set(zip(x, y)) # set of tuples


def make_KDtree(pts_2d):
    return KDTree(pts_2d, leaf_size=1)


def find_angle(origin, axis_pt, other_pt):
    vec1 = np.subtract(axis_pt, origin) # Treat as arbitrary x-axis
    vec2 = np.subtract(other_pt, origin)

    dot_product = np.dot(vec1, vec2)
    v1_norm = np.linalg.norm(vec1)
    v2_norm = np.linalg.norm(vec2)
    # print(f"origin_pt: {origin}, axis_pt: {axis_pt}, other_pt: {other_pt}")
    # print("Input to arccos", dot_product/(v1_norm * v2_norm))
    
    angle = np.arccos(np.clip(dot_product/(v1_norm * v2_norm), -1, 1)) # np.arccos requires input ∈ [-1, 1]
    angle = angle * (180/np.pi) # Radians to degrees, range (0, 180)

    # Determine if vectors are within 180 degrees of each other
    # by looking at if the signs of their y-coordinates match
    if (vec1[1] * vec2[1]) < 0:
        angle = angle + 180
    return angle % 360 # Get angle in [0, 360)

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
        print(f"origin_pt: {origin_pt}, axis_pt: {axis_pt}, pt: {pt}")
        print(f"angle: {angle}")
        hash[math.floor(angle/10)] = 1
    print("hash:", hash)
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

    for i, pt in enumerate(pts_2d):
        if tuple(pt) in alpha_shape:
            continue 
        nn_ind = kdtree.query_radius(pt.reshape(1, 2), r=1)[0] # [array([idx1, ... , idxk])]
        if not is_surrounded360(nn_ind, pts_2d):
            bound_ind.append(i)
    # print("Number of points bounding holes:", len(bound_ind))
    return bound_ind

def see_bounds(bound_ind, pts_2d):
    bounds = pts_2d[bound_ind]
    bounds_x = bounds[:, 0]
    bounds_y = bounds[:, 1]

    x, y = pts_2d.T
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x,y, s=0.1)
    ax1.scatter(bounds_x, bounds_y, s=0.5, color='red')
    plt.show()

def main(args):
    pts_all, pts_2d = original2pcs(args)
    bound_ind = bound_holes(pts_2d)
    see_bounds(bound_ind, pts_2d)
    

def debug():

    find_angle([0.03642289, 0.01385439],
               [0.03673349, 0.01392236],
               [0.03888722, 0.01763664])
    find_angle(np.array([0.03642289, 0.01385439]), 
               np.array([0.03673349, 0.01392236]), 
               np.array([0.04053209, 0.01475373]))


def pcs2original():
    # Need to work with n_components = dimension of original data
    # I.e. Need n_components = 3 to map to 3D
    # Results of mapping 3-dim pcs back to original space may not be EXACTLY same as original points
    # May need to truncate to some number of decimal places
    pass



if __name__ == "__main__":
    args = get_arguments()
    main(args)

    # debug()


    # # Check that alpha_points is contained in pts_2d
    # print(all([pt in pts_2d for pt in alpha_points]))

    # # Check that principal components can be transformed back to original space exactly (up to some number of decimal places)
    # vtcs = read_dotobj(args.obj_path)['v']
    # vtcs_df = pd.DataFrame(vtcs, columns=['x', 'y', 'z'])
    # pca = PCA()
    # pca.fit(vtcs_df)

    # original = pca.inverse_transform(all_pcs[:5])
    # original = [tuple(coords) for coords in original]
    # print(original)
    # print(vtcs[:5])
    # for coords in original:
    #     print(tuple(round(coord, 6) for coord in coords) in vtcs[:5])

    # # Animate the points being plotted one at a time
    # # to see if points adjacent to each other in the array are also nearby spatially
    # # No.
    # plt.ion()
    # xs, ys = pts_2d.T
    # for i in range(len(xs)):
    #     x = xs[i]
    #     y = ys[i]
    #     # plt.gca().cla() # optionally clear axes
    #     plt.scatter(x, y, s=2, color='blue')
    #     plt.title(str(i))
    #     plt.draw()
    #     plt.pause(0.000000000001)

    # plt.show(block=True) # block=True lets the window stay open at the end of the animation.
    
    # # Giving KDtree a try
    # tree = make_KDtree(pts_2d)

    # ind = tree.query_radius(pts_2d[128, :].reshape(1, 2), r=0.0045)
    # for i in range(len(ind[0])):
    #     print(f"Index: {ind[0][i]}")

    # # Testing find_angle with a small example
    # origin = [0, 0]
    # vtx1 = [5, 1]
    # vtx2 = [1, -4.7]
    # angle = find_angle(origin, vtx1, vtx2)
    # print(angle)

    # dist = mean_neighbor_dist(pts_2d, tree)
    # print(dist)
