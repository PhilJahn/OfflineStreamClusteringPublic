from scipy.spatial.distance import euclidean

from birch import Birch
from clustream import CluStream
from denstream import DenStream
from dstream import DStream
from emcstream import EMCStream
from gbfuzzystream import GBFuzzyStream
from hddstream import HDDStream
from stmstream import STMStream
from streamkmeans import STREAMKmeans
from streamkmplus import STREAMKMPlus


def get_online_method(method, config_dict, seed):
    if method == "clustream":
        param_dict = {"max_mc": 100, "init_num": 2000, "mbf": 2.0, "m": 10, "delta": 512,
                      "dist": "euclidean"} | config_dict
        param_dict["b"] = 1
        param_dict["l"] = 0
        param_dict["window"] = None
        method = CluStream(max_mc=param_dict["max_mc"], init_num=param_dict["init_num"],
                           mbf=param_dict["mbf"], m=param_dict["m"], delta=param_dict["delta"],
                           random_state=seed, b=param_dict["b"], l=param_dict["l"], dist=param_dict["dist"],
                           verbose=False)

    elif method == "denstream":
        param_dict = {"init_num": 1000, "mu": 10, "beta": 0.2, "epsilon": 16.0, "b": 2, "l": 0.25, "stream_speed": 1000,
                      "alpha": 0.5, "heuristic": 1, "distance": "euclidean"} | config_dict
        param_dict["heuristic"] = param_dict["heuristic"] == 1
        method = DenStream(init_num=param_dict["init_num"], mu=param_dict["mu"], beta=param_dict["beta"],
                           epsilon= param_dict["epsilon"], b=param_dict["b"], l=param_dict["l"],
                           stream_speed=param_dict["stream_speed"], alpha=param_dict["alpha"],
                           heuristic=param_dict["heuristic"], distance=param_dict["distance"], verbose=False)
    elif method == "hddstream":
        param_dict = {"init_num": 1000, "window": 1000, "mu": 10, "beta": 0.5, "epsilon": 0.2, "delta":0.001, "pi":30, "kappa":1000, "b": 2, "l": 0.5} | config_dict
        method = HDDStream(init_num=param_dict["init_num"], window=param_dict["window"], mu=param_dict["mu"], beta=param_dict["beta"],
                           epsilon=param_dict["epsilon"], delta=param_dict["delta"], pi=param_dict["pi"], kappa=param_dict["kappa"],
                           b=param_dict["b"], l=param_dict["l"], verbose = False)
    elif method == "streamkmeans":
        param_dict = {"k": 100, "chunk_size": 1000, "prototype_num": 500} | config_dict
        method = STREAMKmeans(k=param_dict["k"], chunk_size=param_dict["chunk_size"],
                              prototype_num=param_dict["prototype_num"], random_state=seed)
    elif method == "streamkmplus":
        param_dict = {"m": 100} | config_dict
        method = STREAMKMPlus(m=param_dict["m"], random_state=seed)
    elif method == "dstream":
        param_dict = {"dim_grid_num": 30, "dim": 2, "value_ranges": 1, "cm": 3.0, "cl": 0.8, "window": 1, "l": 0.998,
                      "beta": 0.3} | config_dict
        method = DStream(dim_grid_num=param_dict["dim_grid_num"], dim=param_dict["dim"],
                         value_ranges=param_dict["value_ranges"], cm=param_dict["cm"], cl=param_dict["cl"],
                         window=param_dict["window"], l=param_dict["l"], beta=param_dict["beta"])
    elif method == "emcstream":
        param_dict = {"k": 10, "horizon": 1000, "init_num_factor": 2, "drift_num_factor": 5,
                      "drift_num_factor_change": 1, "drift_inc_delay": 2, "consistency_threshold": 1.0, "consistency_drift": 0.001,
                      "consistency_delay": 4} | config_dict
        method = EMCStream(k=param_dict["k"], horizon=param_dict["horizon"],
                           init_num_factor=param_dict["init_num_factor"],
                           drift_num_factor=param_dict["drift_num_factor"],
                           drift_num_factor_change=param_dict["drift_num_factor_change"],
                           drift_inc_delay=param_dict["drift_inc_delay"],
                           consistency_threshold=param_dict["consistency_threshold"],
                           consistency_drift=param_dict["consistency_drift"],
                           consistency_delay=param_dict["consistency_delay"],
                           random_state=seed, verbose=False)
    elif method == "gbfuzzystream":
        param_dict = {"l": 0.2, "threshold": 0.3, "m": 2, "eps": 10, "dist": "euclidean", "window": 1000} | config_dict
        method = GBFuzzyStream(l=param_dict["l"], threshold=param_dict["threshold"], m=param_dict["m"], eps=param_dict["eps"],
                               dist=param_dict["dist"], window=param_dict["window"])
    elif method == "stmstream":
        param_dict = {"epsilon": 0.1, "k": 10, "window": 1000, "stride": 500, "init_num": 1000, "dist": "euclidean",
                      "u_radius": 0.05 } | config_dict
        method = STMStream(epsilon=param_dict["epsilon"], k=param_dict["k"], window=param_dict["window"], stride=param_dict["stride"],
                           init_num=param_dict["init_num"], dist=param_dict["dist"], u_radius=param_dict["u_radius"],
                           random_state=seed, verbose=False)
    elif method == "birch":
        param_dict = {"max_leaf_entry_num": 100, "L": 4, "B": 2, "closeness": "D0", "absorption": "R", "outlier_ratio": 0.1, "outlier_batch": 100} | config_dict
        method = Birch(max_leaf_entry_num=param_dict["max_leaf_entry_num"], L=param_dict["L"], B=param_dict["B"],
                       closeness=param_dict["closeness"], absorption=param_dict["absorption"],
                       outlier_ratio=param_dict["outlier_ratio"], outlier_batch=param_dict["outlier_batch"])
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return method
