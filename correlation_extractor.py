import argparse

import numpy as np
from scipy import stats

from defines import real_datasets, online_method_names, main_online_algorithms, dataset_names


def get_scores_from_file(filename):
    scores = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i].replace("\n", "")
            linesplit = line.split(",")
            if linesplit[0] not in scores:
                scores[linesplit[0]] = {}
            scores[linesplit[0]][linesplit[1]] = float(linesplit[2])
    return scores

if __name__ == "__main__":
    dss = real_datasets
    dss = np.sort(dss)
    online_methods = main_online_algorithms

    scores_seed_online = get_scores_from_file("result_extracts/scores_seed_online_main_purity.txt")
    scores_kmeans = get_scores_from_file("result_extracts/scores_kmeans_main_offline_score.txt")
    scores_dbscan = get_scores_from_file("result_extracts/scores_dbscan_main_offline_score.txt")
    scores_adjacency = get_scores_from_file("result_extracts/scores_adjacency_main_offline_score.txt")
    scores_dpc = get_scores_from_file("result_extracts/scores_dpc_main_offline_score.txt")



    spearman_table_content = ""
    pearson_table_content = ""

    for ds in dss:
        dsscore_seed_online = []
        dsscore_kmeans = []
        dsscore_dbscan = []
        dsscore_adjacency = []
        dsscore_dpc = []
        for online_method in online_methods:
            online_alg = online_method_names[online_method]
            if online_method == "gbfuzzystream_original" and ds in ["kddcup99", "sensorstream", "rialto", "phiusiil", "smartmeter", "powersupply", "sepsissurvival", "mirs"]:
                continue
            elif online_method == "dstream" and ds in ["heartbeats", "phiusiil", "cancer"]:
                continue
            elif online_method == "clustream_river" and ds in ["cancer", "starlightcurves", "mirs"]:
                continue
            else:
                dsscore_seed_online.append(scores_seed_online[online_alg][ds])
                dsscore_kmeans.append(scores_kmeans[online_alg][ds])
                dsscore_dbscan.append(scores_dbscan[online_alg][ds])
                dsscore_adjacency.append(scores_adjacency[online_alg][ds])
                dsscore_dpc.append(scores_dpc[online_alg][ds])


        def statistic_spearman(x,):  # permute only `x`
            return stats.spearmanr(x, dsscore_seed_online).statistic
        def statistic_pearson(x,):  # permute only `x`
            return stats.pearsonr(x, dsscore_seed_online).statistic

        spear_kmeans = stats.permutation_test((dsscore_kmeans,), statistic_spearman, permutation_type='pairings')
        pears_kmeans = stats.permutation_test((dsscore_kmeans,), statistic_pearson, permutation_type='pairings')
        print(ds, "k-Means:", spear_kmeans.statistic, spear_kmeans.pvalue, pears_kmeans.statistic, pears_kmeans.pvalue)
        spear_dbscan = stats.permutation_test((dsscore_dbscan,), statistic_spearman, permutation_type='pairings')
        pears_dbscan = stats.permutation_test((dsscore_dbscan,), statistic_pearson, permutation_type='pairings')
        print(ds, "DBSCAN:", spear_dbscan.statistic, spear_dbscan.pvalue, pears_dbscan.statistic, pears_dbscan.pvalue)
        spear_dpc = stats.permutation_test((dsscore_dpc,), statistic_spearman, permutation_type='pairings')
        pears_dpc = stats.permutation_test((dsscore_dpc,), statistic_pearson, permutation_type='pairings')
        print(ds, "DPC:", spear_dpc.statistic, spear_dpc.pvalue, pears_dpc.statistic, pears_dpc.pvalue)
        spear_adjacency = stats.permutation_test((dsscore_adjacency,), statistic_spearman, permutation_type='pairings')
        pears_adjacency = stats.permutation_test((dsscore_adjacency,), statistic_pearson, permutation_type='pairings')
        print(ds, "Adjacency:", spear_adjacency.statistic, spear_adjacency.pvalue, pears_adjacency.statistic, pears_adjacency.pvalue)

        spear_kmeans_col = "\\cellcolor{red!25}" if spear_kmeans.pvalue > 0.05 else "\\cellcolor{blue!25}"
        spear_dbscan_col = "\\cellcolor{red!25}" if spear_dbscan.pvalue > 0.05 else "\\cellcolor{blue!25}"
        spear_dpc_col = "\\cellcolor{red!25}" if spear_dpc.pvalue > 0.05 else "\\cellcolor{blue!25}"
        spear_adjacency_col = "\\cellcolor{red!25}" if spear_adjacency.pvalue > 0.05 else "\\cellcolor{blue!25}"

        pears_kmeans_col = "\\cellcolor{red!25}" if pears_kmeans.pvalue > 0.05 else "\\cellcolor{blue!25}"
        pears_dbscan_col = "\\cellcolor{red!25}" if pears_dbscan.pvalue > 0.05 else "\\cellcolor{blue!25}"
        pears_dpc_col = "\\cellcolor{red!25}" if pears_dpc.pvalue > 0.05 else "\\cellcolor{blue!25}"
        pears_adjacency_col = "\\cellcolor{red!25}" if pears_adjacency.pvalue > 0.05 else "\\cellcolor{blue!25}"


        spearman_table_content += (f"{dataset_names[ds]} & {spear_adjacency_col} {spear_adjacency.statistic:.2f} ({spear_adjacency.pvalue:.3f}) & "
                                   f"{spear_kmeans_col} {spear_kmeans.statistic:.2f} ({spear_kmeans.pvalue:.3f}) &"
                                   f"{spear_dbscan_col} {spear_dbscan.statistic:.2f} ({spear_dbscan.pvalue:.3f}) &"
                                   f"{spear_dpc_col} {spear_dpc.statistic:.2f} ({spear_dpc.pvalue:.3f}) &"
                                   f"\\\\\n")

        pearson_table_content += (f"{dataset_names[ds]} & {pears_adjacency_col} {pears_adjacency.statistic:.2f} ({pears_adjacency.pvalue:.3f}) & "
                                   f"{pears_kmeans_col} {pears_kmeans.statistic:.2f} ({pears_kmeans.pvalue:.3f}) &"
                                   f"{pears_dbscan_col} {pears_dbscan.statistic:.2f} ({pears_dbscan.pvalue:.3f}) &"
                                   f"{pears_dpc_col} {pears_dpc.statistic:.2f} ({pears_dpc.pvalue:.3f})"
                                   f"\\\\\n")

    with open("table_spearman.txt", "w") as f:
        f.write(spearman_table_content)
    with open("table_pearson.txt", "w") as f:
        f.write(pearson_table_content)