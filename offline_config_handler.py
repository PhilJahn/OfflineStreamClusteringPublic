import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition

from utils import get_data_properties


def get_offline_configspace(method, data_points, labels, args):
    config_space = ConfigurationSpace()

    d, gt_cls_num = get_data_properties(data_points, labels)

    base_1 = Constant("ds", args.ds)
    base_2 = Constant("online", args.online)
    base_3 = Constant("offline", method)
    base_4 = Constant("batch_size", args.batch)
    base_5 = Constant("batch_max", args.batch_max)
    base_6 = Constant("batch_budget", args.batch_budget)
    base_7 = Constant("batch_id", args.batch_id)
    base_8 = Constant("postprocess", args.postprocess)

    config_space.add([base_1, base_2, base_3, base_4, base_5, base_6, base_7, base_8])

    if method == "dbscan":
        dbscan_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        dbscan_2 = Integer("min_samples", (1, 25), default=5)
        config_space.add([dbscan_1, dbscan_2])
    elif method == "kmeans":
        kmeans_1 = Integer("num_clusters", (1, 100), default=gt_cls_num)
        kmeans_2 = Categorical("init", ["k-means++", "random"], default="k-means++")
        config_space.add([kmeans_1, kmeans_2])
    elif method == "agglomerative":
        agglomerative_1 = Categorical("connectivity", [None, "kneighbors_graph"], default=None)
        agglomerative_2 = Integer("n_neighbors", (1, 100), default=10)
        agglomerative_3 = Categorical("linkage", ["ward", "complete", "average", "single"], default="ward")
        config_space.add([agglomerative_1, agglomerative_2, agglomerative_3])
        agglomerative_cond_1 = EqualsCondition(config_space['n_neighbors'], config_space['connectivity'],
                                               "kneighbors_graph")
        config_space.add(agglomerative_cond_1)
    elif method == "dpc":
        dpc_1 = Constant("metric", "euclidean")
        dpc_2 = Categorical("use_distance_threshold", [True, False], default=False)
        dpc_3 = Float("distance_threshold", (0, (d ** 0.5) / 2), default=0.5)
        dpc_4 = Categorical("gaussian", [True, False], default=True)
        dpc_5 = Categorical("use_min_rho", [True, False], default=False)
        dpc_6 = Float("min_rho", (1, 100), default=5)
        dpc_7 = Categorical("use_min_delta", [True, False], default=False)
        dpc_8 = Float("min_delta", (0, (d ** 0.5) / 2), default=0.5)
        dpc_9 = Categorical("hard_assign", [True, False], default=False)
        dpc_10 = Categorical("halo", [True, False], default=True)
        dpc_11 = Categorical("halo_avg", [True, False], default=True)
        config_space.add([dpc_1, dpc_2, dpc_3, dpc_4, dpc_5, dpc_6, dpc_7, dpc_8, dpc_9, dpc_10, dpc_11])
        dpc_cond_1 = EqualsCondition(config_space['distance_threshold'], config_space['use_distance_threshold'], True)
        dpc_cond_2 = EqualsCondition(config_space['min_rho'], config_space['use_min_rho'], True)
        dpc_cond_3 = EqualsCondition(config_space['min_delta'], config_space['use_min_delta'], True)
        dpc_cond_4 = EqualsCondition(config_space['halo_avg'], config_space['halo'], True)

        config_space.add(dpc_cond_1)
        config_space.add(dpc_cond_2)
        config_space.add(dpc_cond_3)
        config_space.add(dpc_cond_4)
    elif method == "predecon":
        predecon_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        predecon_2 = Integer("min_samples", (1, 25), default=5)
        predecon_3 = Categorical("enforce_max_pref_dimensionality", [True, False], default=False)
        if d-1 > 1:
            predecon_4 = Integer("max_pref_dimensionality", (1, d-1), default=d-1)
        else:
            predecon_4 = Constant("max_pref_dimensionality", 1)
        predecon_5 = Float("variance_threshold", (0, 0.01), default=0.0005) #equivalent to 5 for 100, capped at 100 for 100
        predecon_6 = Integer("kappa", (1,100), default=100)
        config_space.add([predecon_1, predecon_2, predecon_3, predecon_4, predecon_5, predecon_6])
        predecon_cond_1 = EqualsCondition(config_space['max_pref_dimensionality'], config_space['enforce_max_pref_dimensionality'], True)
        config_space.add(predecon_cond_1)
    elif method == "adjacency":
        adjacency_1 = Integer("weight", (0, 50), default=1)
        config_space.add([adjacency_1])
        if args.online in ["clustream", "denstream"]:
            adjacency_2 = Float("range", (1, 10), default=1)
            config_space.add([adjacency_2])

    else:
        raise NotImplementedError(method)
    return config_space