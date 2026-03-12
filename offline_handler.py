import ast
import math
import os

from pandas.core.window import online
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import kneighbors_graph

from datahandler import load_data
from offline.adjacency import get_adjacency_mc
from density_peak_clustering import DensityPeakClustering
from offline.predecon import PreDeCon
import numpy as np
import pandas as pd


def get_summary_clustering(data_points, weights, offline_method, config_dict, seed):
    if offline_method == "dbscan":
        config = {"eps": 0.5, "min_samples": 5} | config_dict
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        clustering = dbscan.fit_predict(data_points, None, sample_weight=weights)
    elif offline_method == "fdbscan":
        config = {"eps": 0.5, "min_samples": 5} | config_dict
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        clustering = dbscan.fit_predict(data_points, None, sample_weight=weights*100)
    elif offline_method == "kmeans":
        config = {"n_clusters": 8, "init": "k-means++"} | config_dict
        kmeans = KMeans(n_clusters=config["n_clusters"], init=config["init"], random_state=seed)
        clustering = kmeans.fit_predict(data_points, None, sample_weight=weights)
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

        print(config)

        dpc = DensityPeakClustering(metric=config["metric"], distance_threshold=config["distance_threshold"],
                                    gaussian=config["gaussian"], min_rho=config["min_rho"],
                                    min_delta=config["min_delta"], hard_assign=config["hard_assign"],
                                    halo=config["halo"], halo_avg=config["halo_avg"],
                                    halo_noise=config["halo_noise"], allow_duplicate=config["allow_duplicate"])
        clustering = dpc.fit_predict(np.array(data_points), None, sample_weight=weights)
    else:
        raise NotImplementedError
    #clustering = get_noise_labels(clustering)
    return clustering


def get_offline_clustering(assignment, summary_clustering):
    summary_clustering_list = np.array(summary_clustering).tolist()
    summary_clustering_list.append(-1)
    offline_clustering = np.array(summary_clustering_list)[assignment]
    #print(summary_clustering_list[:min(len(summary_clustering_list), 50)])
    #print(assignment[:50])
    #print(offline_clustering[:50])
    return offline_clustering.reshape(len(assignment))
