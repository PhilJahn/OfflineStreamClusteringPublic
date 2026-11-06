import argparse
import os

import numpy as np
from scipy import stats

from defines import real_datasets, online_method_names, main_online_algorithms, dataset_names, complete_real_datasets


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


online_metric_files = {"bop_score": "structure_main_real_completed_bop_score",
                       "bop_score_real": "structure_main_real_completed_bop_score_real",
                       "bop_score_summary": "structure_main_real_completed_bop_score_summary",
                       "summary_ssil": "structure_main_real_completed_summary_ssil",
                       "purity": "seed_online_main_real_completed_purity",
                       "offline_score": "seed_online_main_real_completed_offline_score",
                       "adjacency": "adjacency_main_real_completed_offline_score",
                       "summary_chi": "structure_main_real_completed_summary_chi",
                       "clu_num": "seed_online_main_real_completed_clu_num",
                       "batch_ts": "seed_online_main_real_completed_batch_ts",
                       }

offline_metric_files = {"dbscan": "dbscan_main_real_completed_offline_score",
                        "kmeans": "kmeans_main_real_completed_offline_score",
                        "dpc": "dpc_main_real_completed_offline_score",
                        "adjacency": "adjacency_main_real_completed_offline_score",
                        "shade": "shade_main_real_completed_offline_score",
                        }


def extract_correlation(online_metric, offline_metric):
    print(online_metric, offline_metric)
    dss = complete_real_datasets
    dss = np.sort(dss)
    online_methods = main_online_algorithms

    online_metric_file = online_metric_files[online_metric]
    offline_metric_file = offline_metric_files[offline_metric]

    corr_scores_online = get_scores_from_file(f"result_extracts/scores_{online_metric_file}.txt")
    corr_scores_offline = get_scores_from_file(f"result_extracts/scores_{offline_metric_file}.txt")

    content = ""

    for ds in dss:
        ds_scores_online = []
        ds_scores_offline = []
        for online_method in online_methods:
            online_alg = online_method_names[online_method]
            online_score = corr_scores_online[online_alg][ds]
            if online_metric in ["bop_score", "bop_score_real", "bop_score_summary"]:
                online_score = -1 * online_score  # inverted for ranking
            ds_scores_online.append(online_score)
            ds_scores_offline.append(corr_scores_offline[online_alg][ds])

        def statistic_spearman(x, ):  # permute only `x`
            return stats.spearmanr(x, ds_scores_online).statistic

        def statistic_pearson(x, ):  # permute only `x`
            return stats.pearsonr(x, ds_scores_online).statistic

        spear = stats.permutation_test((ds_scores_offline,), statistic_spearman, permutation_type='pairings')
        content += f"{ds},{spear.statistic},{spear.pvalue}\n"

    if not os.path.exists(f"corr_scores"):
        os.makedirs(f"corr_scores", exist_ok=True)

    with open(f"corr_scores/{online_metric}_to_{offline_metric}.txt", "w") as f:
        f.write(content)


if __name__ == "__main__":
    online_metrics = ["summary_chi"] #["clu_num", "batch_ts", "bop_score", "bop_score_real", "bop_score_summary", "summary_ssil", "purity", "offline_score", "adjacency", "summary_chi"]
    offline_metrics = ["shade", "dpc", "dbscan", "kmeans", "adjacency"]
    for online_metric in online_metrics:
        for offline_metric in offline_metrics:
            extract_correlation(online_metric, offline_metric)
    print("done")
