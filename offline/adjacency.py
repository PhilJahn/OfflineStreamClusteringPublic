from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

from utils import dict_to_np, get_noise_labels
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
    #print(radii)

    dense = [True if weight >= config["weight"] else False for weight in weights]
    #print(weights)
    #print(dense)
    #print(np.array(weights)[dense])
    dense_ids = np.arange(len(centers), dtype=int)[dense]
    #print(dense_ids)
    dense_id_inv = np.ones(len(centers), dtype=int) * -1
    j = 0
    for i in range(len(dense)):
        if dense[i]:
            dense_id_inv[i] = j
            j += 1

    #print(dense_id_inv)
    centers = np.array(centers)
    dm = distance_matrix(centers[dense], centers, p=2)
    #print(dm)
    center_clustering = np.ones(len(centers), dtype=int) * -1
    curClu = -1

    for i in range(len(dense_ids)): # no need to iterate over all, since we already know core points from above
        dense_id = dense_ids[i]
        if center_clustering[dense_id] == -1:
            curClu += 1
            center_clustering[dense_id] = curClu
            queue = set()
            inv_i = dense_id_inv[dense_id]
            for j in range(len(dm[inv_i])):
                radius = max(radii[inv_i], radii[j])
                if dm[inv_i][j] <= config["radius"]*radius and inv_i != j:
                    queue.add(j)
            while(len(queue) > 0):
                curi = queue.pop()
                center_clustering[curi] = curClu
                r = []
                inv_cur_i = dense_id_inv[curi]
                if dense[curi]:
                    for j in range(len(dm[inv_cur_i])):
                        radius = max(radii[inv_cur_i], radii[j])
                        if dm[inv_cur_i][j] <= config["radius"] * radius and inv_i != j:
                            r.append(j)
                for x in r:
                    if center_clustering[x] == -1:
                        center_clustering[x] = curClu
                        if dense[x]:
                            queue.add(x)
    #print(center_clustering)
    center_clustering = get_noise_labels(center_clustering)

    # plt.figure(figsize=(8, 8))
    # plt.scatter(centers[:, 0], centers[:, 1], c=center_clustering, s=config["radius"] * np.array(radii)*500, cmap=plt.get_cmap("jet"))
    # for i in range(len(centers)):
    #     plt.text(x=centers[i][0], y=centers[i][1], s=str(i)+":"+str(center_clustering[i]))
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # plt.scatter(centers[:, 0], centers[:, 1], c=dense)
    # plt.show()

    #print(dm.shape)

    clustering = center_clustering[assignments]
    #print("clus", clustering)

    return clustering

