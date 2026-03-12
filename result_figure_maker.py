import argparse
import math
import os
from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from table_maker import smaller_better_metrics

online_method_names = {"full_batch": "Full Batch", "random": "Random", "lwc": "LWC-M", "lwc_square": "LWC", "kcentroid": "k-Centroid", "protras": "ProTraS",
                       "streamkmeans+": "STREAMKMeans", "streamkmeans_river": "StreamKMeans", "streamkmplus+": "StreamKM++", "birch_sklearn": "BIRCH", "clustream": "CluSt. (ours)", "clustream_river": "CluStream",
                       "denstream": "DenSt. (ours)", "denstream_river": "DenStream", "dstream": "D-Stream", "swclustering+": "SWClustering", "gbfuzzystream_original": "GB-FuzzyStream", "gbfuzzystream": "GB-FuzzyStream (ours)",
                       "stmstream+": "STM-Stream", "emcstream": "EMCSt. (ours)", "emcstream_original": "EMCStream", "pure_offline": "Full Data"
                       }

online_colors = {"full_batch": "lightgray", "random": "red", "lwc": "darkred", "lwc_square": "darkred", "kcentroid": "orangered", "protras": "coral",
                       "streamkmeans+": "tan", "streamkmeans_river": "tan", "streamkmplus+": "orange", "birch_sklearn": "blue", "clustream": "darkblue", "clustream_river": "darkblue",
                       "denstream": "lightblue", "denstream_river": "lightblue","swclustering+": "aqua", "dstream": "brown", "gbfuzzystream_original": "deeppink", "gbfuzzystream": "deeppink",
                       "stmstream+": "pink", "emcstream": "darkviolet", "emcstream_original": "darkviolet", "pure_offline": "darkgray"
                       }

def make_result_figure(ds_list, metric, online_list, offline):

    plt.rcParams.update({'font.size': 20})
    ds_list_rank = copy(ds_list)
    ds_list_rank.append("rank")

    result_store = {}
    for ds in ds_list:
        result_store[ds] = {}
        result_store[ds][metric] = {}
        for online in online_list:
            store_name = f"result_evals/{ds}_{online}_{offline}_1000_1800.npy"
            if os.path.exists(store_name):
                results = np.load(store_name, allow_pickle=True).item()
                result_store[ds][metric][online] = results["all"][metric]
            else:
                print(store_name, "is missing")
                result_store[ds][metric][online] = np.nan

    sorted_ds_metric = {}
    for ds in ds_list:
        sorted_ds_metric[ds] = {}
        results = [result_store[ds][metric][online] for online in online_list]
        if metric in smaller_better_metrics:
            results = np.nan_to_num(results, nan=np.inf)
            sorted_ds_metric[ds][metric] = np.sort(results, axis=0)
        else:
            results = np.nan_to_num(results, nan=-np.inf)
            sorted_ds_metric[ds][metric] = np.flip(np.sort(results, axis=0))

    ranks = {}
    for online in online_list:
        ranks[online] = {}
        ranks[online][metric] = {}
        for ds in ds_list:
            sorted_results = sorted_ds_metric[ds][metric]
            result = result_store[ds][metric][online]
            if math.isnan(result):
                if metric in smaller_better_metrics:
                    result = np.inf
                else:
                    result = -np.inf
            rank = sorted_results.tolist().index(result) + 1
            ranks[online][metric][ds] = rank
        ranks[online][metric]["all"] = np.mean(list(ranks[online][metric].values()))

    #color = plt.cm.tab20(np.linspace(0, 1, 20))

    colors = ["blue", "purple", "orange", "turquoise", "red"]

    plt.figure(figsize=(20,2))
    for i in range(len(online_list)):
        online = online_list[i]
        method_ranks = []
        for ds in ds_list:
            method_ranks.append(ranks[online][metric][ds])


        black = 'black'
        plt.boxplot(method_ranks, labels=[online_method_names[online]], positions=[i], widths=[0.5], patch_artist=True, boxprops=dict(facecolor=online_colors[online], color=black),
        capprops=dict(color=black),
        whiskerprops=dict(color=black),
        flierprops=dict(color=black, markeredgecolor=black),
        medianprops=dict(color=black))
    plt.xticks(rotation=20, ha='right')
    #plt.grid()
    plt.savefig(f"result_figures/result_boxplot_{offline}.jpg", bbox_inches='tight', transparent=True)
    plt.savefig(f"result_figures/result_boxplot_{offline}.svg", bbox_inches='tight', transparent=True)

    #plt.legend()
    #plt.show()
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default="paper", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="none", type=str, help='Offline Clustering Method')
    args = parser.parse_args()



    dslist = ["stream_densired", "rbf3", "har", "electricity", "insects_abrupt"]
    if args.metric == "paper":
        if args.offline not in ["online"]:
            metric_list = "offline_score"
        else:
            metric_list = "purity"
    else:
        raise NotImplementedError

    for offline in ["kmeans", "dbscan", "dpc", "adjacency"]:
        if offline != "adjacency":
            online_list = ["pure_offline", "full_batch", "random", "lwc_square", "kcentroid", "protras",
                           "streamkmeans_river", "streamkmplus+", "birch_sklearn", "clustream_river",
                           "denstream_river", "swclustering+", "dstream", "emcstream_original",
                           "gbfuzzystream_original",
                           "stmstream+"]
        else:
            online_list = ["random", "lwc_square", "kcentroid", "protras",
                           "streamkmeans_river", "streamkmplus+", "birch_sklearn", "clustream_river",
                           "denstream_river", "swclustering+", "dstream", "emcstream_original",
                           "gbfuzzystream_original",
                           "stmstream+"]
        make_result_figure(dslist, metric_list, online_list, offline)