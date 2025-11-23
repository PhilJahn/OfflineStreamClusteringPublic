import numpy as np
from matplotlib import pyplot as plt

from datahandler import load_data
from batch_learn import BatchLearn, BatchKMeans
from clustream import CluStream
from online.denstream_river import DenStream
from online.streamkmeans import STREAMKMeans
from utils import dps_to_np, get_noise_labels, dict_to_np
from evaluation_handler import evaluate_clustering
from sklearn.neighbors import BallTree, KDTree


def get_online_method(method, config_dict, seed):
    if method == "clustream_online" or method == "clustream":
        param_dict = {"n_macro_clusters": 10, "mmc": 100, "time_gap": 100000000, "mu": 0.5,
                      "sigma": 0.5, "mc_r_factor": 2.0, "time_window": 1000} | config_dict
        method = CluStream(n_macro_clusters=param_dict["n_macro_clusters"], max_micro_clusters=param_dict["mmc"],
                           micro_cluster_r_factor=param_dict["mc_r_factor"],
                           time_gap=param_dict["time_gap"], time_window=param_dict["time_window"],
                           sigma=param_dict["sigma"],
                           mu=param_dict["mu"],
                           seed=seed)
    elif method == "denstream_online" or method == "denstream":
        param_dict = {"decaying_factor": 0.25, "beta": 0.75, "mu": 2, "epsilon": 0.02, "n_samples_init": 1000,
                      "stream_speed": 100} | config_dict
        method = DenStream(decaying_factor=param_dict["decaying_factor"], beta=param_dict["beta"],
                           mu=param_dict["mu"],
                           epsilon=param_dict["epsilon"], n_samples_init=param_dict["n_samples_init"],
                           stream_speed=param_dict["stream_speed"])
    elif method == "batch":
        method = BatchLearn()
    elif method == "batchkmeans":
        param_dict = {"n_clusters": 8, "init": "k-means++"} | config_dict
        method = BatchKMeans(n_clusters=param_dict["n_clusters"], init=param_dict["init"], seed=seed)
    elif method == "streamkmeans":
        param_dict = {"chunk_size": 10, "n_clusters": 2, "mu": 0.5, "sigma": 0.5} | config_dict
        method = STREAMKMeans(chunk_size=param_dict["chunk_size"], n_clusters=param_dict["n_clusters"],
                              mu=param_dict["mu"], sigma=param_dict["sigma"], seed=seed)
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return method


def get_online_assignments(method, summary, batch):
    if method == "clustream_online" or method == "clustream":
        centers = np.array(summary)[:, 1]
    elif method == "denstream_online" or method == "denstream":
        p_summary, o_summary = summary
        centers = np.array(p_summary)[:, 1]
        labels = np.array(p_summary)[:, 4]
    elif method == "streamkmeans":
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
    if method == "denstream_online" or method == "denstream":
        cur_preds = labels[cur_preds]
    return cur_preds # make sure that id num = order in summary


incremental = ["clustream", "clustream_online", "denstream", "denstream_online", "streamkmeans"]
dict_methods = ["clustream", "clustream_online", "denstream", "denstream_online", "streamkmeans"]


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

    metrics = evaluate_clustering(data_batches=data_batches, label_batches=label_batches, clusterings=clusterings, eval_mode=eval_mode)

    if return_summary:
        return metrics, clusterings, summaries
    else:
        return metrics, clusterings

if __name__ == "__main__":
    config_dict = {}
    config_dict["online"] = "denstream"
    config_dict["ds"] = "complex9"
    config_dict["batch_size"] = 1000
    config_dict["batch_max"] = 3
    config_dict["batch_id"] = 0
    config_dict["batch_budget"] = 30
    config_dict["seed"] = 0
    config_dict["n_clusters"] = 100
    config_dict["epsilon"] = 0.1
    config_dict["beta"] = 0.51
    config_dict["mu"] = 2
    config_dict["decaying_factor"] = 0.001
    config_dict["stream_speed"] = 100
    config_dict["n_samples_init"] = 200
    batch_size = config_dict["batch_size"]
    batch_id = config_dict["batch_id"]
    data_points, labels = load_data("complex9",seed=0, batch=config_dict["batch_size"], batch_max=config_dict["batch_max"])
    metrics, clusterings, summaries = evaluate_online_clustering(data_points, labels, config_dict["online"], config_dict, config_dict["batch_size"], return_summary=True, eval_mode="purity")

    p_mcs, o_mcs = summaries[2]

    centers = []
    lbs = []
    for p_mc in p_mcs:
        #print(p_mc)
        center = dict_to_np(p_mc[1])
        centers.append(center)
        lbs.append(p_mc[4])
    centers = np.array(centers)
    print(np.unique(lbs), np.max(lbs)+1, len(lbs))
    #print(centers)
    print(metrics["purity"])
    plt.figure(figsize=(8, 8))
    plt.scatter(data_points[2000:3000,0], data_points[2000:3000,1], c=clusterings[2],cmap=plt.get_cmap("jet"))
    plt.scatter(centers[:,0], centers[:,1], c=lbs, s=150, linewidths=1, edgecolors= "black", cmap=plt.get_cmap("jet"))
    plt.show()
    plt.close()
    #pprint(summaries)