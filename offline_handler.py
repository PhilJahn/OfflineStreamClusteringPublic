import ast
import math
import os

from pandas.core.window import online
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import kneighbors_graph
from predecon import PreDeCon

from datahandler import load_data
from evaluation_handler import evaluate_clustering
from offline.adjacency import get_adjacency_mc
from offline.density_peak_clustering import DensityPeakClustering
from utils import get_noise_labels, dict_to_np
import numpy as np
import pandas as pd


def get_summary_clustering(data_points, offline_method, config_dict, seed):
    if offline_method == "dbscan":
        config = {"eps": 0.5, "min_samples": 5} | config_dict
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        clustering = dbscan.fit_predict(data_points, None)
    elif offline_method == "kmeans":
        config = {"n_clusters": 8, "init": "k-means++"} | config_dict
        kmeans = KMeans(n_clusters=config["n_clusters"], init=config["init"], random_state=seed)
        clustering = kmeans.fit_predict(data_points, None)
    elif offline_method == "agglomerative":
        config = {"n_clusters": 8, "connectivity": None, "linkage": "ward"} | config_dict
        if config["connectivity"] == "kneighbors_graph":
            config["connectivity"] = kneighbors_graph(data_points, n_neighbors=config["n_neighbors"])
        agglomerative = AgglomerativeClustering(n_clusters=config["n_clusters"], connectivity=config["connectivity"],
                                                linkage=config["linkage"])
        clustering = agglomerative.fit_predict(data_points, None)
    elif offline_method == "dpc":
        config = {"metric": "euclidean", "use_distance_threshold": False, "distance_threshold": None,
                  "gaussian": True, "use_min_rho": False, "min_rho": None, "use_min_delta": False, "min_delta": None,
                  "hard_assign": False, "halo": True, "halo_avg": True, "halo_noise": True
                  } | config_dict
        print(config)
        dpc = DensityPeakClustering(metric=config["metric"], distance_threshold=config["distance_threshold"],
                                    gaussian=config["gaussian"], min_rho=config["min_rho"],
                                    min_delta=config["min_delta"], hard_assign=config["hard_assign"],
                                    halo=config["halo"], halo_avg=config["halo_avg"],
                                    halo_noise=config["halo_noise"])
        clustering = dpc.fit_predict(np.array(data_points), None)
    elif offline_method == "predecon":
        pd_dps = pd.DataFrame.from_records(data_points)
        config = {"eps": 0.5, "min_samples": 5, "enforce_max_pref_dimensionality": False, "max_pref_dimensionality": 1,
                  "variance_threshold": 0.0005, "kappa": 100} | config_dict
        if not config["enforce_max_pref_dimensionality"]:
            config["enforce_max_pref_dimensionality"] = np.inf

        predecon_clusterer = PreDeCon(eps=config["eps"], min_samples=config["min_samples"],
                                      max_dimensions=config["max_pref_dimensionality"],
                                      variance_thres=config["variance_threshold"],
                                      kappa=config["kappa"])
        predecon_clusterer.fit(pd_dps)
        clustering = predecon_clusterer.labels_
    elif offline_method == "adjacency":  # handled by process_summaries
        labels = np.arange(len(data_points)+1)
        negative_labels = -len(data_points)-1 + labels
        labellist = labels.tolist()
        labellist.extend(negative_labels.tolist())
        return np.array(labellist)
    else:
        raise NotImplementedError
    clustering = get_noise_labels(clustering)
    return clustering


def get_offline_clustering(assignment, summary_clustering):
    offline_clustering = summary_clustering[assignment]
    return offline_clustering.reshape(len(assignment))


def load_summaries(ds, online, batch_id, config_dict, load_default=False):
    param_log_file_name = f"online_param_logs/{ds}/online_params_{ds}_{online}_b{config_dict['batch_size']}_m{config_dict['batch_max']}_t{config_dict['batch_budget']}.csv"
    best_summary_path = None
    best_assignment_path = None

    with open(param_log_file_name, "r") as param_log_file:
        best_score = np.inf
        default = True
        for line in param_log_file:
            if (not default and not load_default) or (default and load_default):
                line = line.strip("\n")
                lineSplit = line.split(";")
                score = float(lineSplit[2])
                if score < best_score:
                    best_score = score
                    best_summary_path = lineSplit[-2]
                    best_assignment_path = lineSplit[-1]
                    best_config = ast.literal_eval(lineSplit[1])
            if default:
                default = False
    print(f"best_score: {best_score}")
    print(f"best_summary: {best_summary_path}")
    print(f"best_config: {best_config}")
    summaries = []
    assignments = []
    i = 0
    best_summary_path_full = best_summary_path + f"_j{batch_id}_s{i}.npy"
    best_assignment_path_full = best_assignment_path + f"_j{batch_id}_s{i}.npy"
    while os.path.exists(best_summary_path_full):
        best_summary = np.load(best_summary_path_full, allow_pickle=True)
        summaries.append(best_summary)
        best_assignment = np.load(best_assignment_path_full, allow_pickle=True)
        assignments.append(best_assignment)

        i += 1
        best_summary_path_full = best_summary_path + f"_j{batch_id}_s{i}.npy"
        best_assignment_path_full = best_assignment_path + f"_j{batch_id}_s{i}.npy"
    return summaries, assignments, best_config


def process_summary(summary, assignments, data_points, online, postprocessing, best_config):
    if online == "clustream":
        weights = []
        centers = []
        variances = []
        radii = []
        for mc in summary:
            center = dict_to_np(mc[1])
            centers.append(center)
            weight = int(mc[3])
            weights.append(weight)
            variance = dict_to_np(mc[-1])
            variances.append(variance)
            radius = float(mc[2])
            radii.append(radius)
        if postprocessing is None:
            return [centers], [assignments]
        elif postprocessing == "weighted":
            weights = np.array(weights)
            overall_weight = np.sum(weights)
            weight_ratio = 1000 / overall_weight
            adjusted_weights = np.ceil(weights * weight_ratio)  # use rounded up weights
            dps = []
            mapping = {}
            for i in range(len(adjusted_weights)):
                mapping[i] = len(dps)
                weighted_dp = [centers[i]] * int(adjusted_weights[i])
                dps.extend(weighted_dp)
            #print(mapping)
            #print(adjusted_weights)
            # from https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
            u, inv = np.unique(assignments, return_inverse=True)
            assignments_new = np.vectorize(mapping.get)(u)[inv].reshape(assignments.shape)
            assert (len(np.unique(assignments_new)) == len(np.unique(assignments)))
            _, u_pos = np.unique(dps, axis=0, return_index=True)
            for a in np.unique(assignments_new):
                assert (a in u_pos)
            return [dps], [assignments_new]
        else:
            raise NotImplementedError
    elif online == "batch":
        return [summary], [assignments]
    else:
        raise NotImplementedError

def process_adjacency(summary, assignments, config_dict):
    config = {"weight": 1, "radius": 1} | config_dict
    if config["online"] == "clustream":
        clustering = get_adjacency_mc(summary, assignments, config)
    else:
        raise NotImplementedError

    return [clustering]


def evaluate_offline_clustering(batch_id, method, config_dict, seed=0):
    data_points_full, labels_full = load_data(config_dict["ds"], 0, config_dict["batch_size"], config_dict["batch_max"])
    start = config_dict["batch_size"] * batch_id
    stop = min(config_dict["batch_size"] * (batch_id + 1), len(data_points_full))
    data_batch = data_points_full[start:stop]
    labels_batch = labels_full[start:stop]
    if config_dict["online"] != "batch":
        summaries, assignments, best_config = load_summaries(config_dict["ds"], config_dict["online"], batch_id,
                                                             config_dict)
    else:
        summaries = [data_batch]
        assignments = [np.arange(len(data_batch))]
        best_config = {}
    final_metrics = {"ari": 0.0, "ami": 0.0}
    summary_data_points_list = []
    assignment_list = []
    clusterings = []
    if config_dict["offline"] != "adjacency":
        for i in range(len(summaries)):
            summary_data_points_set, assignments_set = process_summary(summaries[i], assignments[i], data_batch,
                                                                       config_dict["online"],
                                                                       config_dict["postprocess"], best_config)
            for j in range(len(summary_data_points_set)):
                summary_data_points_list.append(summary_data_points_set[j])
                assignment_list.append(assignments_set[j])
    else:
        for i in range(len(summaries)):
            summary_data_points_set, _ = process_summary(summaries[i], assignments[i], data_batch,
                                                                       config_dict["online"],
                                                                       config_dict["postprocess"], best_config)
            assignments_set = process_adjacency(summaries[i], assignments[i], config_dict)
            for j in range(len(summary_data_points_set)):
                summary_data_points_list.append(summary_data_points_set[j])
                assignment_list.append(assignments_set[j])
    for i in range(len(summary_data_points_list)):
        summary_data_points = summary_data_points_list[i]
        assignment = assignment_list[i]
        print(summary_data_points)
        print(assignment)
        summary_clustering = get_summary_clustering(summary_data_points, method, config_dict, seed)
        print(summary_clustering)
        clustering = get_offline_clustering(assignment, summary_clustering)
        clusterings.append(clustering)


        plt.figure(figsize=(8, 8))
        plt.scatter(data_batch[:, 0], data_batch[:, 1], c=clustering)
        if config_dict["online"] != "batch":
            plt.scatter(np.unique(summary_data_points, axis=0)[:, 0], np.unique(summary_data_points, axis=0)[:, 1], s=100,
                        c="black")
        plt.show()

        metrics = evaluate_clustering(data_batches=[data_batch], label_batches=[labels_batch], clusterings=[clustering],
                                      eval_mode="ariami")
        final_metrics["ari"] += metrics["ari"]
        final_metrics["ami"] += metrics["ami"]
    final_metrics["ari"] /= len(summaries)
    final_metrics["ami"] /= len(summaries)
    return final_metrics, clusterings


import matplotlib.pyplot as plt

if __name__ == "__main__":
    config_dict = {}
    config_dict["online"] = "clustream"
    config_dict["offline"] = "adjacency"
    config_dict["ds"] = "densired2"
    config_dict["batch_size"] = 1000
    config_dict["batch_max"] = 3
    config_dict["postprocess"] = None
    config_dict["batch_id"] = 0
    config_dict["batch_budget"] = 30
    batch_size = config_dict["batch_size"]
    batch_id = config_dict["batch_id"]
    evaluate_offline_clustering(batch_id, config_dict["offline"], config_dict)

    # summaries, assignments, _ = load_summaries("densired2", "clustream", config_dict["batch_id"], config_dict, False)
    # # summary = summaries[0]
    # # assignment = assignments[0]
    # print(len(np.unique(assignments[0])))
    # summary_data_points_set, assignments_set = process_summary(summaries[0], assignments[0], data_points,
    #                                                            config_dict["online"], config_dict["postprocess"],
    #                                                            {})
    #
    # print(len(np.unique(assignments_set[0])))
    #
    # # print(summary)
    # # print(assignment)
    # # centers = []
    # # for mc in summary:
    # #     center = dict_to_np(mc[1])
    # #     centers.append(center)
    # # centers = np.array(centers)
    # centers = np.array(summary_data_points_set[0])
    # assignment = assignments_set[0]
    #
    # _, ui = np.unique(centers, return_index=True, axis=0)
    # print(np.sort(np.unique(assignment)))
    #
    # plt.figure(figsize=(8,8))
    # plt.scatter(batch_data_points[:,0], batch_data_points[:,1], c=batch_labels)
    # plt.scatter(centers[:,0], centers[:,1], c="black", s=100)
    # plt.show()
    #
    # plt.figure(figsize=(8,8))
    # plt.scatter(batch_data_points[:,0], batch_data_points[:,1], c=assignment)
    # plt.scatter(centers[:,0], centers[:,1], c="black", s=100)
    # plt.show()
    #
    # plt.figure(figsize=(8,8))
    # plt.scatter(batch_data_points[:,0], batch_data_points[:,1], c=assignment)
    # plt.scatter(np.unique(centers, axis=0)[:,0], np.unique(centers, axis=0)[:,1], s=100, c=ui.tolist())
    # plt.show()
