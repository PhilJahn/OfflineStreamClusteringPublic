import argparse
import os
from timeit import default_timer

from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade

import numpy as np

from offline_config_handler import get_offline_configspace

global method
global supervised
global data_points
global labels
global performance_log_file


def clustering_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    start = default_timer()
    clustering = perform_clustering(data_points, method, config_dict, seed)
    time = float(default_timer() - start)
    metrics = eval_clustering_supervised(clustering, labels)
    metrics |= eval_clustering_unsupervised(clustering, data_points)
    metrics |= {"clu_num": len(np.unique(clustering)), "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
    if supervised:
        score = 2 - metrics["AMI"] - metrics["ARI"]
    else:
        score = 2 - metrics["SilhouetteScore"] - metrics["DISCO5"]
    performance_log_file.write(f"{config_dict}, {metrics}\n")
    return score



def run_parameter_estimation(args, seed, name):
    scenario = Scenario(get_offline_configspace(method, args.ds, data_points), use_default_config=True,
                        n_trials=10000000, walltime_limit=args.budget, seed=seed, name=name)
    smac = HyperparameterOptimizationFacade(scenario, clustering_runner, overwrite=True)
    incumbent = smac.optimize()
    score = smac.runhistory.get_min_cost(incumbent)
    run_num = len(smac.runhistory.items())
    return incumbent, score, scenario, run_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="densired2", type=str, help='Dataset')
    parser.add_argument('--online', default="kmeans", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="dbscan", type=str, help='Offline Clustering Method')
    parser.add_argument('--budget', default=600, type=int, help='SMAC AutoML Budget (in seconds)')
    args = parser.parse_args()



