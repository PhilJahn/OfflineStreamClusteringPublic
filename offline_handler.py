import ast
import math
import os
import time

from pandas.core.window import online
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import kneighbors_graph

import utils
from datahandler import load_data
from density_peak_clustering import DensityPeakClustering
#from offline.predecon import PreDeCon
import numpy as np
import pandas as pd


def get_adjacency_clustering(positions, radii, status):
    mc_dists = utils.distance_calc(positions, positions, dist="euclidean")
    mc_dists = np.subtract(mc_dists, radii)
    mc_dists = mc_dists.T
    mc_dists = np.subtract(mc_dists, radii)
    connected = np.where(mc_dists <= 0, True, False)
    partitions = np.arange(len(positions) + 1)  # get connected
    print(status)
    for i in range(len(partitions) - 1):
        if status[i] != "outlie" and status[i] != "sporad" and status[i] != "sporadic" and status[i] != "deleted" and \
                status[i] != "delete":
            for j in range(i):
                if status[j] != "outlie" and status[j] != "sporad" and status[j] != "sporadic" and status[
                    j] != "deleted" and status[j] != "delete":
                    if connected[i, j]:
                        if partitions[i] != partitions[j]:
                            if status[i] == "dense" and status[j] == "dense":
                                key = min(partitions[i], partitions[j])
                                other_key = max(partitions[i], partitions[j])
                                partitions[np.where(partitions == other_key)[0]] = key
                            elif status[i] == "normal" and status[j] == "dense":
                                partitions[i] = partitions[j]
                            elif status[i] == "dense" and status[j] == "normal":
                                partitions[j] = partitions[i]
        else:
            partitions[i] = -1
    partitions[-1] = -1
    return partitions




def get_summary_clustering(data_points, radii, weights, status, offline_method, config_dict, seed):
    if offline_method == "dbscan":
        config = {"eps": 0.5, "min_samples": 5} | config_dict
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        start = time.time()
        clustering = dbscan.fit_predict(data_points, None, sample_weight=weights)
        stop = time.time()
    elif offline_method == "fdbscan":
        config = {"eps": 0.5, "min_samples": 5} | config_dict
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        start = time.time()
        clustering = dbscan.fit_predict(data_points, None, sample_weight=weights*100)
        stop = time.time()
    elif offline_method == "kmeans":
        config = {"n_clusters": 8, "init": "k-means++"} | config_dict
        kmeans = KMeans(n_clusters=config["n_clusters"], init=config["init"], random_state=seed)
        start = time.time()
        clustering = kmeans.fit_predict(data_points, None, sample_weight=weights)
        stop = time.time()
    elif offline_method == "dpc":
        config = {"metric": "euclidean", "distance_threshold": None,
                  "gaussian": True, "min_rho": None, "min_delta": None,
                  "hard_assign": False, "halo": True, "halo_avg": True,
                  "halo_noise": True, "allow_duplicate": False
                  } | config_dict
        #print(config)
        config["gaussian"] = bool(config["gaussian"])
        config["hard_assign"] = bool(config["hard_assign"])
        config["halo"] = bool(config["halo"])
        config["halo_avg"] = bool(config["halo_avg"])
        config["halo_noise"] = bool(config["halo_noise"])
        config["allow_duplicate"] = bool(config["allow_duplicate"])

        #print(config)
        start = time.time()
        dpc = DensityPeakClustering(metric=config["metric"], distance_threshold=config["distance_threshold"],
                                    gaussian=config["gaussian"], min_rho=config["min_rho"],
                                    min_delta=config["min_delta"], hard_assign=config["hard_assign"],
                                    halo=config["halo"], halo_avg=config["halo_avg"],
                                    halo_noise=config["halo_noise"], allow_duplicate=config["allow_duplicate"])
        stop = time.time()
        clustering = dpc.fit_predict(np.array(data_points), None, sample_weight=weights)
    elif offline_method == "adjacency":
        start = time.time()
        clustering = get_adjacency_clustering(data_points, radii, status)
        stop = time.time()
    else:
        raise NotImplementedError
    #clustering = get_noise_labels(clustering)
    return clustering, stop - start


def get_offline_clustering(assignment, summary_clustering):
    summary_clustering_list = np.array(summary_clustering).tolist()
    summary_clustering_list.append(-1)
    offline_clustering = np.array(summary_clustering_list)[assignment]
    #print(summary_clustering_list[:min(len(summary_clustering_list), 50)])
    #print(assignment[:50])
    #print(offline_clustering[:50])
    return offline_clustering.reshape(len(assignment))
