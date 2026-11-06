import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition
from kneed import KneeLocator

import utils
from datahandler import load_data
from density_peak_clustering import get_distance_matrix
from utils import get_data_properties

import kneed


def get_offline_configspace(method, summary_centers, weights, labels, args):
    config_space = ConfigurationSpace()

    d, gt_cls_num = get_data_properties(summary_centers, labels)

    base_1 = Constant("ds", args.ds)
    base_2 = Constant("online", args.online)
    base_3 = Constant("offline", method)
    base_4 = Constant("batch_size", args.batch)
    base_5 = Constant("eval_size", args.eval_size)
    base_6 = Constant("offline_batch_budget", args.budget)
    base_7 = Constant("online_batch_budget", args.online_batch_budget)

    base_8 = Constant("batch_id", args.batch_id)

    config_space.add([base_1, base_2, base_3, base_4, base_5, base_6, base_7, base_8])

    if method == "dbscan":
        max_min_samples = min(50, len(summary_centers)) # range based on Anna Beer, Ekaterina Kuznetsova, Claudia Plant: Parameter Selection for DBSCAN: Insights from Persistent Homology. ECAI 2025: 3250-3257
        min_samples, eps, max_eps = estimate_defaults_dbscan(summary_centers, "euclidean", max_min_samples)
        if max_eps >= 0.000001:
            dbscan_1 = Float("eps", (0, max_eps), default=eps)
        else:
            dbscan_1 = Categorical("eps", [0, 0.000001], default=0.000001)
        dbscan_2 = Integer("min_samples", (1, max_min_samples), default=min_samples)
        config_space.add([dbscan_1, dbscan_2])
    elif method == "fdbscan":
        max_min_samples = min(50, len(summary_centers)) # range based on Anna Beer, Ekaterina Kuznetsova, Claudia Plant: Parameter Selection for DBSCAN: Insights from Persistent Homology. ECAI 2025: 3250-3257
        min_samples, eps, max_eps = estimate_defaults_dbscan(summary_centers, "euclidean", max_min_samples)
        if max_eps >= 0.000001:
            fdbscan_1 = Float("eps", (0, max_eps), default=eps)
        else:
            fdbscan_1 = Categorical("eps", [0, 0.000001], default=0.000001)
        fdbscan_2 = Integer("min_samples", (1, 100*max_min_samples), default=100*min_samples)
        config_space.add([fdbscan_1, fdbscan_2])
    elif method == "kmeans":
        max_num_clusters = min(100, len(summary_centers))
        if max_num_clusters == 1:
            kmeans_1 = Constant("n_clusters", 1)
        else:
            kmeans_1 = Integer("n_clusters", (1, max_num_clusters), default=min(max_num_clusters,gt_cls_num))
        kmeans_2 = Categorical("init", ["k-means++", "random"], default="k-means++")
        config_space.add([kmeans_1, kmeans_2])
    # elif method == "agglomerative":
    #     agglomerative_1 = Categorical("connectivity", [None, "kneighbors_graph"], default=None)
    #     agglomerative_2 = Integer("n_neighbors", (1, 100), default=10)
    #     agglomerative_3 = Categorical("linkage", ["ward", "complete", "average", "single"], default="ward")
    #     config_space.add([agglomerative_1, agglomerative_2, agglomerative_3])
    #     agglomerative_cond_1 = EqualsCondition(config_space['n_neighbors'], config_space['connectivity'],
    #                                            "kneighbors_graph")
    #     config_space.add(agglomerative_cond_1)
    elif method == "dpc":
        max_min_rho = min(100, len(summary_centers))
        distance_threshold, min_rho, max_distance = estimate_defaults_dpc(summary_centers, weights, default_ratio=0.01, dist_metric="euclidean")

        dpc_1 = Constant("metric", "euclidean")
        if max_distance < 0.000001:
            dpc_2 = Constant("distance_threshold", 0)
        else:
            dpc_2 = Float("distance_threshold", (0, max_distance), default=min(distance_threshold, max_distance))
        dpc_3 = Categorical("gaussian", [True, False], default=True)
        if args.online not in ["denstream", "denstream_river", "denstream_river_postnorm", "dstream"]:
            dpc_4 = Float("min_rho", (1, max_min_rho), default=max(1, min(min_rho, max_min_rho)))
        else:
            dpc_4 = Float("min_rho", (0, max_min_rho), default=max(0, min(min_rho, max_min_rho)))
        if max_distance < 0.000001:
            dpc_5 = Constant("min_delta", 0)
        else:
            dpc_5 = Float("min_delta", (0, max_distance), default=min(distance_threshold, max_distance))
        dpc_6 = Categorical("hard_assign", [True, False], default=False)
        dpc_7 = Categorical("halo", [True, False], default=True)
        dpc_8 = Categorical("halo_avg", [True, False], default=True)
        dpc_9 = Categorical("allow_duplicate", [True, False], default=False)
        config_space.add([dpc_1, dpc_2, dpc_3, dpc_4, dpc_5, dpc_6, dpc_7, dpc_8, dpc_9])
        dpc_cond_1 = EqualsCondition(config_space['halo_avg'], config_space['halo'], True)

        config_space.add(dpc_cond_1)
    # elif method == "predecon":
    #     predecon_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
    #     predecon_2 = Integer("min_samples", (1, 25), default=5)
    #     predecon_3 = Categorical("enforce_max_pref_dimensionality", [True, False], default=False)
    #     if d-1 > 1:
    #         predecon_4 = Integer("max_pref_dimensionality", (1, d-1), default=d-1)
    #     else:
    #         predecon_4 = Constant("max_pref_dimensionality", 1)
    #     predecon_5 = Float("variance_threshold", (0, 0.01), default=0.0005) #equivalent to 5 for 100, capped at 100 for 100
    #     predecon_6 = Integer("kappa", (1,100), default=100)
    #     config_space.add([predecon_1, predecon_2, predecon_3, predecon_4, predecon_5, predecon_6])
    #     predecon_cond_1 = EqualsCondition(config_space['max_pref_dimensionality'], config_space['enforce_max_pref_dimensionality'], True)
    #     config_space.add(predecon_cond_1)
    # elif method == "adjacency":
    #     adjacency_1 = Integer("weight", (0, 50), default=1)
    #     config_space.add([adjacency_1])
    #     if args.online in ["clustream", "denstream"]:
    #         adjacency_2 = Float("range", (1, 10), default=1)
    #         config_space.add([adjacency_2])
    elif method == "idec":
        max_num_clusters = min(100, len(summary_centers))
        if max_num_clusters == 1:
            idec_1 = Constant("n_clusters", 1)
        else:
            idec_1 = Integer("n_clusters", (1, max_num_clusters), default=min(max_num_clusters,gt_cls_num))
        idec_2 = Float("alpha", (0.1, 2), default=1)
        idec_3 = Float("clustering_loss_weight ", (0.1, 2), default=1)
        idec_4 = Float("plr", (1e-5, 1e-1), default=1e-3, log=True)
        idec_5 = Float("clr", (1e-5, 1e-1), default=1e-3, log=True)
        idec_6 = Constant("device", "cpu")
        config_space.add([idec_1, idec_2, idec_3, idec_4, idec_5, idec_6])
    elif method == "shade":
        shade_1 = Integer("min_points", (2, 100), default=5)
        shade_2 = Float("plr", (1e-5, 1e-1), default=1e-3, log=True)
        shade_3 = Float("clr", (1e-5, 1e-1), default=1e-3, log=True)
        shade_4 = Constant("device", "cpu")
        config_space.add([shade_1, shade_2, shade_3, shade_4])
    elif method != "adjacency":
        raise NotImplementedError(method)
    return config_space

import matplotlib.pyplot as plt
def estimate_defaults_dbscan(data, dist_metric="euclidean", max_minpts =50):
    d = len(data[0])
    # heuristic from Density-Based Clustering in Spatial Databases: The Algorithm GDBSCAN and its Applications
    default_min_pts = min(max_minpts, d * 2)

    distance_matrix = utils.distance_calc(data, data, dist_metric)
    if len(data) > default_min_pts:
        knn_dist = np.partition(distance_matrix, default_min_pts, axis=0)[default_min_pts]
    else:
        knn_dist = np.max(distance_matrix, axis=0)
    #print(knn_dist)
    #print(knn_dist.shape)
    sorted_knn_dist = np.flip(np.sort(knn_dist))

    default_eps = KneeLocator(range(len(knn_dist)), sorted_knn_dist, S=1.0, curve="convex", direction="decreasing").knee_y
    # plt.figure(figsize=(10, 10))
    # plt.plot(sorted_knn_dist)
    # plt.hlines(y=default_eps, xmin=0, xmax=len(knn_dist))
    # plt.show()
    max_eps = np.max(distance_matrix)
    return default_min_pts, default_eps, max_eps


def estimate_defaults_dpc(data, weights, default_ratio=0.01, dist_metric="euclidean"):
    distance_matrix = get_distance_matrix(data, metric=dist_metric)
    max_distance = np.max(distance_matrix)
    min_rho = default_ratio * np.sum(weights)
    min_weight = min_rho * np.sum(weights)
    weights_vec = weights.reshape(len(weights), 1)
    weight_square = np.dot(weights_vec, weights_vec.T)
    weight_square_flat = weight_square.flatten()
    order = np.argsort(distance_matrix.flatten())
    weights_sorted_by_dist = weight_square_flat[order]
    cur_sum = 0
    i = -1
    while i < len(weights_sorted_by_dist):
        if cur_sum >= min_weight:
            break
        i += 1
        cur_sum += weights_sorted_by_dist[i]

    distances_sorted = distance_matrix.flatten()[order]
    return distances_sorted[i], min_rho, max_distance

