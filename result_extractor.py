import math
import os

import numpy as np

from defines import real_datasets, main_online_algorithms, full_online_algorithms, smaller_better_metrics, \
    online_method_names, metric_names
from critdd import Diagram
import pandas as pd

def extract_results_to_file(offline, online_key, online_list, ds_list, metric):
    result_store = {}

    substitution_ds = []
    for ds in ds_list:
        result_store[ds] = {}
        result_store[ds] = {}
        for online in online_list:
            if offline == "online":
                suffix = "1000_300_logdim"
                store_name = f"result_evals/{offline}/{ds}_{online}_{suffix}.npy"
            elif offline == "seed_online":
                suffix = "1000_300_logdim_600_logdim"
                store_name = f"result_evals/{offline}/{ds}_{online}_seed_online_{suffix}.npy"
            elif offline == "structure":
                suffix = "1000_300_logdim"
                store_name = f"summary_evals/{ds}_{online}_{suffix}.npy"
            elif online == "pure_offline":
                suffix = "1000_900_logdim_600_logdim"
                store_name = f"result_evals/{offline}/{ds}_{online}_{offline}_{suffix}.npy"
            else:
                suffix = "1000_300_logdim_600_logdim"
                store_name = f"result_evals/{offline}/{ds}_{online}_{offline}_{suffix}.npy"
            if os.path.exists(store_name):
                results = np.load(store_name, allow_pickle=True).item()
                if metric == "offline_score":
                    result_store[ds][online] = 2 - results["all"][metric]
                else:
                    result_store[ds][online] = results["all"][metric]
            else:
                print(store_name, "is missing")
                result_store[ds][online] = np.nan
                if ds not in substitution_ds:
                    substitution_ds.append(ds)

    sorted_ds_results = {}
    for ds in ds_list:
        sorted_ds_results[ds] = {}
        results = [result_store[ds][online] for online in online_list]
        sorted_ds_results[ds] = np.sort(results, axis=0)
        if ds in substitution_ds:
            min_val = sorted_ds_results[ds][0]
            sub_scores = np.load(f"result_evals/online/{ds}_one_1000_300_logdim.npy", allow_pickle=True).item()
            if metric == "offline_score":
                sub_val = 2 - sub_scores["all"][metric]
            elif metric == "summary_ssil":
                sub_val = -2
            else:
                sub_val = sub_scores["all"][metric]
            print(ds, min_val, sub_val)
            results = np.nan_to_num(results, nan=sub_val)
            sorted_ds_results[ds] = np.sort(results, axis=0)
            for online in online_list:
                if math.isnan(result_store[ds][online]):
                    result_store[ds][online] = sub_val






    #print(result_store)
    
    if not os.path.exists("result_extracts/"):
        os.makedirs("result_extracts/", exist_ok=True)

    score_text = f"online_method,dataset_name,{metric_names[metric]}\n"
    for online in online_list:
        online_name = online_method_names[online]
        for ds in ds_list:
            score_text += f"{online_name},{ds},{result_store[ds][online]}\n"

    output_file_name = f"result_extracts/scores_{offline}_{online_key}_{metric}.txt"
    with open(output_file_name, "w") as f:
        f.write(score_text)

    # from https://github.com/mirkobunse/critdd example
    df = pd.read_csv(output_file_name).pivot(
        index="dataset_name",
        columns="online_method",
        values=metric_names[metric]
    )
    diagram = Diagram(
        df.to_numpy(),
        treatment_names=df.columns,
        maximize_outcome=True
    )

    #print(diagram.treatment_names)
    #print(diagram.average_ranks)


    ranks = {}
    for online in online_list:
        ranks[online] = {}
        rank_sum = 0
        for ds in ds_list:
            sorted_results = np.flip(sorted_ds_results[ds])
            result = result_store[ds][online]
            #rank = sorted_results.tolist().index(result) + 1
            # https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
            rank = np.where(sorted_results == result)[0]
            rank = np.mean(rank + 1) # Average of rank on ties as in Friedman Test
            ranks[online][ds] = rank
            rank_sum += rank
        rank_avg = rank_sum / len(ds_list)
        pos_cd = np.where(diagram.treatment_names == online_method_names[online])[0][0]
        rank_cd = diagram.average_ranks[pos_cd]
        print(online, rank_avg, rank_cd)

    ranks_text = f"online_method,dataset_name,rank_{metric_names[metric]}\n"
    for online in online_list:
        online_name = online_method_names[online]
        for ds in ds_list:
            ranks_text += f"{online_name},{ds},{ranks[online][ds]}\n"

    output_file_name = f"result_extracts/ranks_{offline}_{online_key}_{metric}.txt"
    with open(output_file_name, "w") as f:
        f.write(ranks_text)


    #print(sorted_ds_results)
    if not os.path.exists("cd_figures/"):
        os.makedirs("cd_figures/", exist_ok=True)

    diagram.to_file(
        f"cd_figures/{offline}_{online_key}_{metric}_cd.pdf",
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": f"{metric_names[metric]}",
                      "ticklabel style": "anchor=south, yshift=1.3*\\pgfkeysvalueof{/pgfplots/major tick length}, font=\\large",
                        "title style": "yshift =\\baselineskip, font = \\large"},
        tikzpicture_options = {"treatment label/.style": "font=\\large"}
    )

    os.remove(f"cd_figures/{offline}_{online_key}_{metric}_cd.aux")
    os.remove(f"cd_figures/{offline}_{online_key}_{metric}_cd.tex")
    os.remove(f"cd_figures/{offline}_{online_key}_{metric}_cd.log")


if __name__ == '__main__':
    offline_algs = ["online", "seed_online", "kmeans", "dbscan", "dpc", "adjacency"]
    online_algss = ["main", "full"]

    extract_results_to_file("structure", "main", main_online_algorithms, real_datasets, "summary_ssil")

    for online_algs in online_algss:
        if online_algs == "main":
            online_alg_list = main_online_algorithms
        else:
            online_alg_list = full_online_algorithms
        for offline_alg in offline_algs:
            if offline_alg == "online" or offline_alg == "seed_online":
                metrics = ["purity", "batch_ts"]
            else:
                metrics = ["offline_score", "batch_ts"]

            if (offline_alg != "online" and offline_alg != "seed_online") or online_algs != "full":
                for metric in metrics:
                    extract_results_to_file(offline_alg, online_algs, online_alg_list, real_datasets, metric)
