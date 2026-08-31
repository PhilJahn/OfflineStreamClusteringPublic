import numpy as np
from scipy.spatial.distance import euclidean

#from birch import Birch
#from clustream import CluStream
#from denstream import DenStream
from dstream import DStream
#from emcstream import EMCStream
#from gbfuzzystream import GBFuzzyStream
#from hddstream import HDDStream
from original_implementations.EmCStream import OriginalEmcStream
from original_implementations.birch_sklearn import SklearnBirch
from original_implementations.gbfuzzystream.MBStream import OriginalGBFuzzyStream
from river_local.cluster.river_clustream import RiverCluStream
from river_local.cluster.river_denstream import RiverDenStream
from river_local.cluster.river_streamkmeans import RiverSTREAMKMeans
from stmstream import STMStream
#from streamkmeans import STREAMKmeans
from streamkmplus import STREAMKMPlus
from subset_methods import BatchSummarizer, WindowSummarizer
from swclustering import SWClustering

non_deterministic_online = [
    "batch_random", "batch_kcentroid", "batch_lwc_square",
    "window_random", "window_lwc_square", "window_kcentroid", "window_protras",
    "emcstream", "stmstream", "streamkmplus", "streamkmplus+",
                            "streamkmeans", "streamkmeans+",# "clustream",
                            "streamkmeans_river", "emcstream_original"]


def get_online_method(method, config_dict, seed):

    # elif method == "denstream":
    #     param_dict = {"init_num": 1000, "mu": 10, "beta": 0.2, "epsilon": 16.0, "b": 2, "l": 0.25, "stream_speed": 1000,
    #                   "alpha": 0.5, "heuristic": 1, "dist": "euclidean"} | config_dict
    #     param_dict["heuristic"] = param_dict["heuristic"] == 1
    #     method = DenStream(init_num=param_dict["init_num"], mu=param_dict["mu"], beta=param_dict["beta"],
    #                        epsilon=param_dict["epsilon"], b=param_dict["b"], l=param_dict["l"],
    #                        stream_speed=param_dict["stream_speed"], alpha=param_dict["alpha"],
    #                        heuristic=param_dict["heuristic"], distance=param_dict["dist"], verbose=False)
    # elif method == "hddstream":
    #     param_dict = {"init_num": 1000, "window": 1000, "mu": 10, "beta": 0.5, "epsilon": 0.2, "delta": 0.001, "pi": 30,
    #                   "kappa": 1000, "b": 2, "l": 0.5} | config_dict
    #     method = HDDStream(init_num=param_dict["init_num"], window=param_dict["window"], mu=param_dict["mu"],
    #                        beta=param_dict["beta"],
    #                        epsilon=param_dict["epsilon"], delta=param_dict["delta"], pi=param_dict["pi"],
    #                        kappa=param_dict["kappa"],
    #                        b=param_dict["b"], l=param_dict["l"], verbose=False)
    # elif method == "streamkmeans" or method == "streamkmeans+":
    #     param_dict = {"k": 100, "chunk_size": 1000, "prototype_num": 500} | config_dict
    #     method = STREAMKmeans(k=param_dict["k"], chunk_size=param_dict["chunk_size"],
    #                           prototype_num=param_dict["prototype_num"], random_state=seed)
    if method == "streamkmplus" or method == "streamkmplus+":
        param_dict = {"m": 100} | config_dict
        method = STREAMKMPlus(m=param_dict["m"], random_state=seed)
    # elif method == "clustream":
    #     param_dict = {"mmc": 100, "init_num": 100, "mbf": 2.0, "m": 10, "delta": 512,
    #                   "dist": "euclidean"} | config_dict
    #     param_dict["b"] = 1
    #     param_dict["l"] = 0
    #     method = CluStream(max_mc=param_dict["mmc"], init_num=param_dict["mmc"],
    #                        mbf=param_dict["mbf"], m=param_dict["m"], delta=param_dict["delta"],
    #                        random_state=seed, b=param_dict["b"], l=param_dict["l"], dist=param_dict["dist"],
    #                        verbose=False)
    elif method == "dstream":
        param_dict = {"dim_grid_num": 30, "dim": 2, "value_ranges": 1, "cm": 3.0, "cl": 0.8, "window": 1, "l": 0.998,
                      "beta": 0.3} | config_dict
        method = DStream(dim_grid_num=param_dict["dim_grid_num"], dim=param_dict["dim"],
                         value_ranges=param_dict["value_ranges"], cm=param_dict["cm"], cl=param_dict["cl"],
                         window=param_dict["window"], l=1 - param_dict["l"], beta=param_dict["beta"])
    elif method == "emcstream":
        param_dict = {"k": 10, "horizon": 1000, "init_num_factor": 2, "drift_num_factor": 5,
                      "drift_num_factor_change": 1, "drift_inc_delay": 2, "consistency_threshold": 1.0,
                      "consistency_drift": 0.001,
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
    # elif method == "gbfuzzystream":

    #     param_dict = {"l": 0.2, "threshold": 0.3, "m": 2, "eps": 10, "dist": "euclidean", "window": 1000,
    #                   "min_split_size": 8} | config_dict
    #     method = GBFuzzyStream(l=param_dict["l"], threshold=param_dict["threshold"], m=param_dict["m"],
    #                            eps=param_dict["eps"], min_split_size=param_dict["min_split_size"],
    #                            dist=param_dict["dist"], window=param_dict["window"])
    elif method == "stmstream":
        param_dict = {"epsilon": 0.1, "k": 10, "window": 1000, "stride": 500, "init_num": 1000, "dist": "euclidean",
                      "u_radius": 0.05} | config_dict
        method = STMStream(epsilon=param_dict["epsilon"], k=param_dict["k"], window=param_dict["window"],
                           stride=param_dict["stride"],
                           init_num=param_dict["init_num"], dist=param_dict["dist"], u_radius=param_dict["u_radius"],
                           random_state=seed, verbose=False)
    # elif method == "birch":
    #     param_dict = {"max_leaf_entry_num": 100, "L": 4, "B": 2, "closeness": "D0", "absorption": "R",
    #                   "outlier_ratio": 0.1, "outlier_batch": 100} | config_dict
    #     method = Birch(max_leaf_entry_num=param_dict["max_leaf_entry_num"], L=param_dict["L"], B=param_dict["B"],
    #                    closeness=param_dict["closeness"], absorption=param_dict["absorption"],
    #                    outlier_ratio=param_dict["outlier_ratio"], outlier_batch=param_dict["outlier_batch"])
    elif method == "swclustering" or method == "swclustering+":
        param_dict = {"mmc": 100, "epsilon": 0.1, "N": 10, "beta": 2, "distance": "euclidean"} | config_dict
        method = SWClustering(max_ehcf_num=param_dict["mmc"], epsilon=param_dict["epsilon"], N=param_dict["N"]*1000,
                              beta=param_dict["beta"], distance=param_dict["distance"])
    elif method in ["batch_random", "batch_kcentroid", "batch_lwc_square", "batch_protras"]:
        param_dict = {"sampling_size": 100, "distance": "euclidean"} | config_dict
        method = BatchSummarizer(sampler=method, count=param_dict["sampling_size"],
                                 distance_function=param_dict["distance"], random_state=seed)
    elif method in ["window_random", "window_kcentroid", "window_lwc_square", "window_protras"]:
        param_dict = {"sampling_size": 100, "distance": "euclidean", "window_size": 1000, "selection_chance": 0.5, "stride": 1000} | config_dict
        method = WindowSummarizer(sampler=method, count=param_dict["sampling_size"], window_size=param_dict["window_size"],
                                  selection_chance=param_dict["selection_chance"], stride=param_dict["stride"],
                                  distance_function=param_dict["distance"], random_state=seed)
    elif method == "full_batch":
        method = BatchSummarizer(sampler=method, count=1000)
    elif method == "clustream_river":
        param_dict = {"mmc": 100, "mbf": 2.0, "window": 1000} | config_dict
        # disable time_gap as only used to compute river clusters, which is done by the offline handler instead
        method = RiverCluStream(max_micro_clusters=param_dict["mmc"], micro_cluster_r_factor=param_dict["mbf"],
                                time_window=param_dict["window"], time_gap=100000000)
    elif method == "denstream_river":
        param_dict = {"init_num": 1000, "mu": 10, "beta": 0.2, "epsilon": 16.0, "l": 0.25,
                      "stream_speed": 1000} | config_dict
        method = RiverDenStream(decaying_factor=param_dict["l"], beta=param_dict["beta"], epsilon=param_dict["epsilon"],
                                mu=param_dict["mu"], stream_speed=param_dict["stream_speed"],
                                n_samples_init=param_dict["init_num"])

    elif method == "streamkmeans_river":
        param_dict = {"chunk_size": 1000, "prototype_num": 500, "mu": 0.5, "sigma": 0.5} | config_dict
        method = RiverSTREAMKMeans(n_clusters=param_dict["prototype_num"], chunk_size=param_dict["chunk_size"],
                                   mu=param_dict["mu"], sigma=param_dict["sigma"], seed=seed)
    elif method == "emcstream_original":
        param_dict = {"k": 10, "horizon": 1000, "consistency_threshold": 1.0, "consistency_drift": 0.001,
                      "init_num_factor": 2, "drift_num_factor": 5, "drift_inc_delay": 4,
                      "consistency_delay": 4} | config_dict

        method = OriginalEmcStream(k=param_dict["k"], horizon=param_dict["horizon"],
                                   ari_threshold=param_dict["consistency_threshold"],
                                   ari_threshold_step=param_dict["consistency_drift"],
                                   init_num_factor=param_dict["init_num_factor"],
                                   drift_num_factor=param_dict["drift_num_factor"],
                                   drift_inc_delay=param_dict["drift_inc_delay"],
                                   consistency_delay=param_dict["consistency_delay"], seed=seed)

    elif method == "emcstream_mod":
        param_dict = {"k": 10, "horizon": 1000, "consistency_threshold": 1.0, "consistency_drift": 0.001,
                      "init_num_factor": 2, "drift_num_factor": 5, "drift_inc_delay": 4,
                      "consistency_delay": 4} | config_dict

        method = ModEmcStream(k=param_dict["k"], horizon=param_dict["horizon"],
                                   ari_threshold=param_dict["consistency_threshold"],
                                   ari_threshold_step=param_dict["consistency_drift"],
                                   init_num_factor=param_dict["init_num_factor"],
                                   drift_num_factor=param_dict["drift_num_factor"],
                                   drift_inc_delay=param_dict["drift_inc_delay"],
                                   consistency_delay=param_dict["consistency_delay"], seed=seed)
    elif method == "gbfuzzystream_original":

        param_dict = {"l": 0.2, "threshold": 0.3, "m": 2, "eps": 10, "dist": "euclidean", "window": 1000,
                      "batch_size": 1000} | config_dict
        method = OriginalGBFuzzyStream(lam=param_dict["l"], threshold=param_dict["threshold"], m=param_dict["m"],
                                       eps=param_dict["eps"], window=param_dict["window"],
                                       batchsize=param_dict["batch_size"])

    elif method == "birch_sklearn":
        param_dict = {"max_leaf_entry_num": 100, "threshold": 0.5, "branching_factor": 50, "batch_size": 1000} | config_dict
        method = SklearnBirch(threshold=param_dict["threshold"], branching_factor=param_dict["branching_factor"],
                              max_summaries=param_dict["max_leaf_entry_num"], batchsize=param_dict["batch_size"],
                              compute_labels=False)

    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return method
