import argparse
import math
import os
import warnings
from copy import copy

import numpy as np
import scipy

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)

smaller_better_metrics = ["davies_bouldin_score", "offline_score", "online_score"]
smaller_better_metrics_noninverted = ["davies_bouldin_score"]

metric_short_names = {"ari": "ARI", "accuracy": "Acc.", "ami": "AMI", "purity": "Pur.", "batch_ts_avg": "TS",
                      "offline_score": "ACI",
                      "cmm": "CMM", "cmm_alt": "CMM+", "fowlkes_mallows": "FMI", "nmi": "NMI",
                      "homogeneity": "HOM", "completeness": "COM", "fair_nmi": "FNMI", "variation_of_information": "VI",
                      "davies_bouldin_score": "DBI", "calinski_harabasz_score": "CH", "disco": "DISCO"
                      }

metric_long_names = {"ari": "ARI", "accuracy": "Clustering accuracy", "ami": "AMI", "purity": "Purity", "batch_ts_avg": "TS",
                     "offline_score": "Sum of ARI and AMI (ACI)",

                     "cmm": "CMM", "cmm_alt": "CMM+", "fowlkes_mallows": "FMI", "nmi": "NMI",
                     "homogeneity": "HOM", "completeness": "COM", "fair_nmi": "FNMI", "variation_of_information": "VI",
                     "davies_bouldin_score": "DBI", "calinski_harabasz_score": "CH", "disco": "DISCO"
                     }

dataset_names = {"stream_densired": "SDENS", "rbf3": "RBF-3", "har": "HAR", "electricity": "Elec", "insects_abrupt": "InsA", "insects_incremental": "InsI", "gassensor": "GasS", "rank": "Avg. Rank"}

online_method_names = {"full_batch": "Full Batch", "random": "Random", "lwc": "LWC-M", "lwc_square": "LWC", "kcentroid": "k-Centroid", "protras": "ProTraS",
                       "streamkmeans+": "StreamKMeans (ours)", "streamkmeans_river": "StreamKMeans", "streamkmplus+": "StreamKM++", "birch_sklearn": "BIRCH", "clustream": "CluStream (ours)", "clustream_river": "CluStream",
                       "denstream": "DenSt. (ours)", "denstream_river": "DenStream", "dstream": "D-Stream", "swclustering+": "SWClustering", "gbfuzzystream_original": "GB-FuzzyStream", "gbfuzzystream": "GB-FuzzyStream",
                       "stmstream+": "STM-Stream", "emcstream": "EMCStream (ours)", "emcstream_original": "EMCStream", "pure_offline": "Full Data"
                       }

offline_method_description = {"dbscan": "for DBSCAN", "kmeans": "for $k$-Means", "dpc": "for DPC", "pure_online": "without any post-processing of the summaries",
                              "adjacency": "according to the purity", "none": "with adjacency post-processing", "online": "for the online phase"
                              }

offline_optimization = {"dbscan": "according to the sum of ARI and AMI", "dpc": "according to the sum of ARI and AMI", "kmeans": "according to the sum of ARI and AMI", "pure_online": "according to the sum of ARI and AMI",
                        "adjacency": "according to the purity after adjacency processing", "none": "according to the purity after adjacency processing", "online": "according to the purity after adjacency processing"
                        }

def make_table(ds_list, metric_list, online_list, offline, appendix):

    #if not appendix:
    while "break" in online_list:
        online_list.remove("break")

    description = "\caption{"
    for metric in metric_list[:-1]:
        description += metric_long_names[metric] + " and "
    description += metric_long_names[metric_list[-1]]
    if not ("davies_bouldin_score" in metric_list or "calinski_harabasz_score" in metric_list or "noise_ratio" in metric_list or "clu_num" in metric_list or "clu_dist" in metric_list) and offline == "online":
        description += " (in \\%)"
    if offline == "online":
        description += " using the individual summaries without adjacency processing. Results correspond to the "
    else:
        description += " results"
    if offline == "adjacency":
        description += " for adjacency processing "
    if not ("davies_bouldin_score" in metric_list or "calinski_harabasz_score" in metric_list or "noise_ratio" in metric_list or "clu_num" in metric_list or "clu_dist" in metric_list) and offline != "online":
        description += " (in \\%)"
    #if offline == "adjacency":
    #    description += " for the evaluated datasets "
    if offline != "online":
        description += " for the evaluated datasets for best-performing hyperparameters "
    description += offline_method_description[offline] + " " + offline_optimization[offline]
    #if not ("davies_bouldin_score" in metric_list or "noise_ratio" in metric_list or "clu_num" in metric_list or "clu_dist" in metric_list) and offline != "online":
    #    description += " (in \\%)"
    description += "."
    if not ("clu_num" in metric_list or "clu_dist" in metric_list):
        description += " The best scores are marked as \\textbf{bold}, and the second-best scores are \\underline{underlined}. A lower rank is better."
    if offline in ["none", "online"]:
        description += " The scores for non-deterministic online phase methods are averaged over $5$ seeds."
    elif offline == "kmeans":
        description += " The scores for $k$-Means are averaged over $5$ seeds."

    if len(metric_list) == 1:
        description += " We report the standard-deviation for "
        if offline == "online":
            description += " the non-deterministic online phase summarizations and "
        description += " the average rank."


    if appendix:
        table = "\\begin{table*}[h]\n\\centering\n"
    else:
        table = "\\begin{table*}[tb]\n\\centering\n"
    table += description

    table += "}\n\\label{tab:results_"
    for metric in metric_list:
        table += metric + "_"
    table += offline
    table += "}\n"

    if appendix and "calinski_harabasz_score" in metric_list:
        table += "\\resizebox{\\textwidth}{!}{\n"

    table += "\\begin{tabular}{|l |"

    metric_header = ""
    data_header = "Summary "
    ds_list_rank = copy(ds_list)
    ds_list_rank.append("rank")

    for dsi in range(len(ds_list_rank)):
        if len(metric_list) > 1:
            for metric in metric_list:
                metric_header += " & " + metric_short_names[metric]
                table += " c |"
            data_header += "& \multicolumn{"
            data_header += str(len(metric_list))
            data_header += "}{c|"
            if dsi != len(ds_list_rank) - 1:
                table += "|"
                data_header += "|"
            data_header += "}{"
            data_header += dataset_names[ds_list_rank[dsi]]
            data_header += "}"
        else:
            data_header += f"& {dataset_names[ds_list_rank[dsi]]}"
            table += "c |"
    table += "} \\hline\n"
    table += data_header
    if metric_header != "":
        table += "\\\\\\hline\n"
        table += metric_header
    table += "\\\\\\hline\hline\n"


    result_store = {}
    std_store = {}
    for ds in ds_list:
        result_store[ds] = {}
        std_store[ds] = {}
        for metric in metric_list:
            result_store[ds][metric] = {}
            std_store[ds][metric] = {}
        for online in online_list:

            if (online == "denstream_river" or online == "dstream") and offline == "dbscan":
                store_name = f"result_evals/{ds}_{online}_fdbscan_1000_1800.npy"
            else:
                store_name = f"result_evals/{ds}_{online}_{offline}_1000_1800.npy"
            if os.path.exists(store_name):
                results = np.load(store_name, allow_pickle=True).item()
                for metric in metric_list:
                    result_store[ds][metric][online] = round(results["all"][metric], 3)
                    std_store[ds][metric][online] = round(results["all_std"][metric], 3)
            else:
                print(store_name, "is missing")
                for metric in metric_list:
                    result_store[ds][metric][online] = np.nan
                    std_store[ds][metric][online] = np.nan

    sorted_ds_metric = {}
    for ds in ds_list:
        sorted_ds_metric[ds] = {}
        for metric in metric_list:
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
        for metric in metric_list:
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
            ranks[online][metric]["all_std"] = np.std(list(ranks[online][metric].values()))

    for ds in ds_list:
        for metric in metric_list:
            if metric in smaller_better_metrics:
                best_score = np.inf
                second_best_score = np.inf
                worst_score = -np.inf
                comp_dir = "less"
            else:
                best_score = -np.inf
                second_best_score = -np.inf
                worst_score = np.inf
                comp_dir = "more"
            for value in result_store[ds][metric].values():
                if comp_dir == "less":
                    if value <= best_score:
                        second_best_score = best_score
                        best_score = value
                    elif value <= second_best_score:
                        second_best_score = value
                    if value > worst_score:
                        worst_score = value
                else:
                    if value >= best_score:
                        second_best_score = best_score
                        best_score = value
                    elif value >= second_best_score:
                        second_best_score = value
                    if value < worst_score:
                        worst_score = value
            result_store[ds][metric]["best_score"] = best_score
            result_store[ds][metric]["second_best_score"] = second_best_score
            result_store[ds][metric]["worst_score"] = worst_score

    for online in online_list:
        if online == "break":
            table = table[:-1]
            table += "\\hline\n"
        else:
            table += online_method_names[online] + " "
            for ds in ds_list:
                for metric in metric_list:
                    result = result_store[ds][metric][online]
                    if math.isnan(result):
                        table += "& -\\ "
                    else:
                        table += "& "
                        best_score = result_store[ds][metric]["best_score"]
                        second_best_score = result_store[ds][metric]["second_best_score"]
                        worst_score = result_store[ds][metric]["worst_score"]
                        #print(ds, metric)
                        random_score = result_store[ds][metric]["random"]

                        if math.isnan(random_score):
                            random_score = worst_score

                        #print(online, ds, metric, best_score, second_best_score, worst_score, random_score)

                        if metric in ["offline_score", "online_score"]:
                            result = 2 - result
                            best_score = 2 - best_score
                            worst_score = 2 - worst_score
                            second_best_score = 2 - second_best_score
                            random_score = 2 - random_score



                        random_best_diff = max(abs(best_score - random_score), 0.00001)
                        random_worst_diff = max(abs(random_score - worst_score), 0.00001)


                        need_closure = False
                        if metric not in ["clu_num", "clu_dist"]:
                            worse_than_random = False
                            random_diff = abs(result - random_score)
                            if metric in smaller_better_metrics_noninverted:
                                if result > random_score:
                                    worse_than_random = True
                                    scale = random_diff/random_worst_diff
                                else:
                                    scale = random_diff/random_best_diff
                            else:
                                if result < random_score:
                                    worse_than_random = True
                                    scale = random_diff / random_worst_diff
                                else:
                                    scale = random_diff / random_best_diff


                            if worse_than_random:
                                table += "\\cellcolor{red!"
                            else:
                                table += "\\cellcolor{blue!"

                            table += str(round(scale*50))
                            table += "}"
                            if result == best_score:
                                table += "\\textbf{"
                                need_closure = True
                            elif result == second_best_score:
                                table += "\\underline{"
                                need_closure = True

                        #if metric == "offline_score":
                        #    if result == 2.0:
                        #        table += f"100"
                        #    else:
                        #        table += f"{result*50:.1f}"
                        if metric not in ["davies_bouldin_score", "calinski_harabasz_score", "noise_ratio", "clu_num", "clu_dist"]:
                            if result == 1.0 and metric != "offline_score":
                                table += f"100"
                            elif result < 0:
                                table += f"{result*100:.0f}"
                            else:
                                table += f"{result*100:.1f}"
                            if len(metric_list) == 1 and offline == "online":
                                result_std = std_store[ds][metric][online]
                                if not math.isnan(result_std):
                                    table += f" $\\pm$ {result_std*100:.1f}"
                        else:
                            table += f"{result}"
                            if len(metric_list) == 1 and offline == "online":
                                 result_std = std_store[ds][metric][online]
                                 if not math.isnan(result_std):
                                     table += f" $\\pm$ {result_std:.1f}"
                        if need_closure:
                            table += "}"
                        table += "\\ "
            for metric in metric_list:
                table += " & "
                own_rank = ranks[online][metric]['all']

                random_rank = ranks["random"][metric]['all']
                if own_rank > random_rank:
                    table += "\\cellcolor{red!"
                else:
                    table += "\\cellcolor{blue!"
                scale = abs(own_rank-random_rank)/len(online_list)
                table += str(round(scale * 50))
                table += "}"


                table += f"{ranks[online][metric]['all']}"

                if len(metric_list) == 1:
                    rank_std = ranks[online][metric]['all_std']
                    table += f"$\\pm$ {rank_std:.1f}"
            table += "\\\\\\hline\n"





    table += "\\end{tabular}\n"
    if appendix and "calinski_harabasz_score" in metric_list:
        table += "}\n"
    table += "\\end{table*}"



    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default="appendix_main", type=str, help='Online Summarization Method')
    #parser.add_argument('--offline', default="online", type=str, help='Offline Clustering Method')
    args = parser.parse_args()

    offline_list = ["online", "kmeans", "dbscan", "dpc", "adjacency"]
    for args.offline in offline_list:
        if args.variant == "paper":
            if args.offline not in ["online"]:
                metric_lists = [["offline_score", "batch_ts_avg"]]
            else:
                metric_lists = [["purity", "batch_ts_avg"]]
        elif args.variant == "appendix_main":
            metric_lists = [["offline_score", "batch_ts_avg"]]
        elif args.variant == "appendix":
            metric_lists = [["ari"], ["ami"], ["purity"], ["cmm"], ["cmm_alt"], ["nmi"],
                            ["accuracy"],
                            ["batch_ts_avg"], ["davies_bouldin_score"], ["calinski_harabasz_score"], ["disco"]
                            ]

        online_list = []

        print(metric_lists)


        if args.offline in ["dbscan", "kmeans", "dpc"]:
            online_list = ["pure_offline", "full_batch", "break", "random", "lwc_square", "kcentroid", "protras", "break",
                           "streamkmeans_river", "streamkmplus+", "break", "birch_sklearn", "clustream_river", "denstream_river", "swclustering+", "break",
                           "dstream", "break", "emcstream_original", "gbfuzzystream_original", "stmstream+"
                           ]
        elif args.offline == "online":
            online_list = ["random", "lwc_square", "kcentroid", "protras", "break",
                           "streamkmeans_river", "streamkmplus+", "break", "birch_sklearn", "clustream_river", "denstream_river", "swclustering+", "break",
                           "dstream", "break", "gbfuzzystream_original", "stmstream+"
                           ]
        elif args.offline == "none":
            online_list = ["full_batch", "break", "random", "lwc_square", "kcentroid", "protras", "break",
                           "streamkmeans_river", "streamkmplus+", "break", "birch_sklearn", "clustream_river", "denstream_river", "swclustering+", "break",
                           "dstream", "break", "gbfuzzystream_original", "stmstream+"
                           ]
        elif args.offline == "adjacency":
            online_list = ["random", "lwc_square", "kcentroid", "protras", "break",
                           "streamkmeans_river", "streamkmplus+", "break", "birch_sklearn", "clustream_river", "denstream_river", "swclustering+", "break",
                           "dstream", "break", "emcstream_original", "gbfuzzystream_original", "stmstream+"
                           ]

        dslist = ["stream_densired", "rbf3", "har", "electricity", "insects_abrupt"]#, "insects_incremental"]

        with open(f"table_{args.offline}_{args.variant}.tex", "w") as f:
            for metric_list in metric_lists:
                table = make_table(dslist, metric_list, online_list, args.offline, args.variant != "paper")
                f.write(table)
            if args.variant != "paper":
                f.write("\n\\clearpage\n")