import numpy as np

from clustpy.metrics import purity, unsupervised_clustering_accuracy
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

from cmm import get_cmm
from online.batch_learn import BatchLearn
from online.clustream import CluStream
from utils import dps_to_np, get_noise_labels
from sklearn.neighbors import BallTree, KDTree

incremental = ["clustream", "clustream_online"]


def get_method(method, config_dict):
    if method == "clustream_online" or method == "clustream":
        param_dict = {"n_macro_clusters": 10, "mmc": 100, "time_gap": 100000000, "mu": 0.5,
                      "sigma": 0.5} | config_dict
        method = CluStream(n_macro_clusters=param_dict["n_macro_clusters"], max_micro_clusters=param_dict["mmc"],
                           micro_cluster_r_factor=param_dict["mc_r_factor"],
                           time_gap=param_dict["time_gap"], time_window=param_dict["time_window"],
                           sigma=param_dict["sigma"],
                           mu=param_dict["mu"],
                           seed=param_dict["seed"])
    elif method == "batch":
        method = BatchLearn()
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return method


def get_assignments(method, summary, batch):
    if method == "clustream_online" or method == "clustream":
        centers = summary[:, 1]
    else:
        centers = summary
    if len(batch[0]) > 3:
        nnTree = BallTree(centers)
    else:
        nnTree = KDTree(centers)
    cur_preds = nnTree.query(batch, 1)[1]
    return cur_preds


dict_methods = ["clustream", "clustream_online"]


def get_clustering(data_points, method, config_dict, batch_size, seed, return_summary):
    method = get_method(method, config_dict)
    if method in dict_methods:
        data_points = dps_to_np(data_points)
    i = 0
    cur_batch = []
    summaries = []
    preds = []
    for dp in data_points:
        if method in incremental:
            method.learn_one(dp)
        cur_batch.append(dp)
        if len(cur_batch) == batch_size:
            if method not in incremental:
                method.learn(cur_batch)
            summary = method.get_summary()
            summaries.append(summary)
            cur_preds = get_assignments(method, summary, cur_batch)
            preds.append(cur_preds)
            cur_batch = []
        i += 1
    if len(cur_batch) > 0:
        summary = method.get_summary()
        summaries.append(summary)
        cur_preds = get_assignments(method, summary, cur_batch)
        preds.append(cur_preds)

    if return_summary:
        return preds, summaries
    else:
        return preds


def evaluate_online_clustering(data_points, labels, method, config_dict, batch_size, seed, return_summary=False,
                               eval_mode="all"):
    clusterings = get_clustering(data_points, method, config_dict, batch_size, seed, return_summary)
    if return_summary:
        clusterings, summaries = clusterings
    labels = get_noise_labels(labels)
    batch_idxs = range(batch_size, len(labels), batch_size)
    label_batches = np.split(labels, batch_idxs)
    data_batches = np.split(data_points, batch_idxs)
    i = 0

    metrics = {"purity":0.0, "accuracy":0.0, "ari":0.0, "nmi":0.0, "ami": 0.0, "cmm": 0.0, "cmm_alt": 0.0}

    for i in range(len(clusterings)):
        cur_labels = label_batches[i]
        print(cur_labels)
        cur_preds = clusterings[i].reshape(len(cur_labels))
        cur_preds = get_noise_labels(cur_preds)
        if eval_mode == "purity":
            cur_purity = purity(cur_labels, cur_preds)
            metrics["purity"] += cur_purity*len(cur_labels)
        elif eval_mode == "ariami":
            cur_ami = adjusted_mutual_info_score(cur_labels, cur_preds)
            metrics["ami"] += cur_ami * len(cur_labels)
            cur_ari = adjusted_rand_score(cur_labels, cur_preds)
            metrics["ari"] += cur_ari*len(cur_labels)
        else:
            cur_purity = purity(cur_labels, cur_preds)
            metrics["purity"] += cur_purity*len(cur_labels)
            cur_acc = unsupervised_clustering_accuracy(cur_labels, cur_preds)
            metrics["accuracy"] += cur_acc*len(cur_labels)
            cur_ami = adjusted_mutual_info_score(cur_labels, cur_preds)
            metrics["ami"] += cur_ami * len(cur_labels)
            cur_nmi = normalized_mutual_info_score(cur_labels, cur_preds)
            metrics["nmi"] += cur_nmi * len(cur_labels)
            cur_ari = adjusted_rand_score(cur_labels, cur_preds)
            metrics["ari"] += cur_ari * len(cur_labels)
            cur_data = data_batches[i]
            cur_cmm, cur_cmm_alt = get_cmm(cur_data, cur_labels, cur_preds, np.ones(len(cur_labels)), k=5, alternate=True)
            metrics["cmm"]  += cur_cmm * len(cur_labels)
            metrics["cmm_alt"] += cur_cmm_alt * len(cur_labels)
    for metric in metrics.keys():
        metrics[metric] = metrics[metric] / len(labels)

    if return_summary:
        return metrics, summaries
    else:
        return metrics
