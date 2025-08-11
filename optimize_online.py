import argparse
from timeit import default_timer

from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade

import numpy as np

from clustpy.metrics import purity
from datahandler import load_data
from online_config_handler import get_online_configspace
from online_handler import evaluate_online_clustering

global online
global data_points
global labels
global batch_size

def online_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    metrics = evaluate_online_clustering(data_points, labels, online, config_dict, batch_size, seed, return_summary=False, eval_mode="purity")
    score = 1-metrics["purity"]
    return score

def run_parameter_estimation(args, seed, name):
    scenario = Scenario(get_online_configspace(args.online, args.batch, data_points, labels), use_default_config=True,
                        n_trials=10000000, walltime_limit=args.budget, seed=seed, name=name)
    smac = HyperparameterOptimizationFacade(scenario, online_runner, overwrite=True)
    incumbent = smac.optimize()
    score = smac.runhistory.get_min_cost(incumbent)
    run_num = len(smac.runhistory.items())
    return incumbent, score, scenario, run_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="densired2", type=str, help='Dataset')
    parser.add_argument('--online', default="kmeans", type=str, help='Online Summarization Method')
    parser.add_argument('--budget', default=600, type=int, help='SMAC AutoML Budget (in seconds)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0)
    batch = args.batch_size
    online = args.online

