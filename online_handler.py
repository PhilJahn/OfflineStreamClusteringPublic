import numpy as np
from online.batch_learn import BatchLearn
from online.clustream import CluStream
from utils import dps_to_np, get_noise_labels
from evaluation_handler import evaluate_clustering
from sklearn.neighbors import BallTree, KDTree


def get_online_method(method, config_dict, seed):
    if method == "clustream_online" or method == "clustream":
        param_dict = {"n_macro_clusters": 10, "mmc": 100, "time_gap": 100000000, "mu": 0.5,
                      "sigma": 0.5, "mc_r_factor": 2.0} | config_dict
        method = CluStream(n_macro_clusters=param_dict["n_macro_clusters"], max_micro_clusters=param_dict["mmc"],
                           micro_cluster_r_factor=param_dict["mc_r_factor"],
                           time_gap=param_dict["time_gap"], time_window=param_dict["time_window"],
                           sigma=param_dict["sigma"],
                           mu=param_dict["mu"],
                           seed=seed)
    elif method == "batch":
        method = BatchLearn()
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return method


def get_online_assignments(method, summary, batch):
    if method == "clustream_online" or method == "clustream":
        centers = np.array(summary)[:, 1]
    else:
        centers = summary
    if method in dict_methods:
        centers = dps_to_np(centers)
    if len(batch[0]) > 3:
        nnTree = BallTree(centers)
    else:
        nnTree = KDTree(centers)
    cur_preds = nnTree.query(batch, 1)[1]
    return cur_preds # make sure that id num = order in summary


incremental = ["clustream", "clustream_online"]
dict_methods = ["clustream", "clustream_online"]


def get_online_clustering(data_points, method_name, config_dict, batch_size, return_summary, seed):
    online_method = get_online_method(method_name, config_dict, seed)
    i = 0
    cur_batch = []
    summaries = []
    preds = []
    for x in data_points:
        if method_name in dict_methods:
            dp = dict(enumerate(x))
        else:
            dp = x
        if method_name in incremental:
            online_method.learn_one(dp)
        cur_batch.append(x)
        if len(cur_batch) == batch_size:
            if method_name not in incremental:
                online_method.learn(cur_batch)
            summary = online_method.get_summary()
            summaries.append(summary)
            cur_preds = get_online_assignments(method_name, summary, cur_batch)
            preds.append(cur_preds)
            cur_batch = []
        i += 1
    if len(cur_batch) > 0:
        summary = online_method.get_summary()
        summaries.append(summary)
        cur_preds = get_online_assignments(method_name, summary, cur_batch)
        preds.append(cur_preds)

    if return_summary:
        return preds, summaries
    else:
        return preds


def evaluate_online_clustering(data_points, labels, method, config_dict, batch_size, return_summary=False,
                               eval_mode="all", seed=0):
    clusterings = get_online_clustering(data_points, method, config_dict, batch_size, return_summary, seed)
    if return_summary:
        clusterings, summaries = clusterings
    labels = get_noise_labels(labels)
    batch_idxs = range(batch_size, len(labels), batch_size)
    label_batches = np.split(labels, batch_idxs)
    data_batches = np.split(data_points, batch_idxs)

    metrics = evaluate_clustering(data_batches=data_batches, label_batches=label_batches, clusterings=clusterings, data_points=data_points, labels=labels, eval_mode=eval_mode)

    if return_summary:
        return metrics, clusterings, summaries
    else:
        return metrics, clusterings
