from scipy.spatial import distance_matrix

from utils import dict_to_np
import numpy as np

def get_adjacency_mc(summary, assignments, config):
    # adjacency == touching radii for MCs above size
    weights = []
    centers = []
    variances = []
    radii = []
    for mc in summary:
        center = dict_to_np(mc[1])
        centers.append(center)
        weight = int(mc[3])
        weights.append(weight)
        radius = float(mc[2])
        radii.append(radius)

    dense = [1 if weight >= config["weight"] else 0 for weight in weights]
    centers = np.array(centers)
    dm = distance_matrix(centers[dense], centers, p=2)
    center_clustering = np.arange(len(centers), dtype=int)

    clustering = center_clustering[assignments]

    return clustering

