import argparse
import math
import os
import traceback
from timeit import default_timer

from ConfigSpace import Configuration
from pynisher import WallTimeoutException
from smac import Scenario, AlgorithmConfigurationFacade

import numpy as np
from torch.utils.data import non_deterministic

from datahandler import load_data
from offline_handler import evaluate_offline_clustering
from online_config_handler import get_online_configspace
from online_handler import evaluate_online_clustering
from time_handler import time_limit

global opt_log_file

global batch_data_points
global batch_labels
global smac

non_deterministic_online = ["emc"]

def online_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    score = np.inf
    online_method = config["online_method"]
    metricss = []
    summariess = []
    scores = []
    try:
        with time_limit(remaining_time):
            if online_method in non_deterministic_online:
                online_seeds = [0,1,2,3,4]
            else:
                online_seeds = [0]
            for seed in online_seeds:
                metrics = evaluate_offline_clustering(config["batch_id"], batch_labels, config["offline_method"], config, seed=seed)
                metricss.append(metrics)
                scores.append(2-metrics["ari"]-metrics["ami"])
            score = np.mean(scores)
    except WallTimeoutException:
        print("Timed out!", flush=True)
        score = np.inf
        raise WallTimeoutException("Timed out!")
    except Exception:
        print(traceback.format_exc(), flush=True)
        score = np.inf
    finally:
        incumbent = smac.intensifier.get_incumbent()
        if incumbent is None:
            cur_best_score = np.inf
        else:
            cur_best_score = smac.runhistory.get_min_cost(incumbent)
        prefix = "-"
        if score < cur_best_score:
            cur_id = len(smac.runhistory)
            prefix = f"clusterings/{args.ds}/{args.online}/{args.name}_{cur_id}"


        opt_log_file.write(f"{config_dict};{score};{prefix}\n")

        for h in range(len(metricss)):
            purity_dict = {"purity": metricss[h]['purity']}
            opt_log_file.write(f"-\t;{h};{purity_dict};\n")
        print(f"{config_dict}: {score}", flush=True)


    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="densired2", type=str, help='Dataset')
    parser.add_argument('--online', default="clustream", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="kmeans", type=str, help='Offline Clustering Method')
    parser.add_argument('--postprocess', default=None, type=str, help='Postprocessing Method')
    parser.add_argument('--budget', default=30
                        , type=int, help='Online SMAC AutoML Budget (in seconds per batch), Offline duget is 2x Online')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--batch_id', default=-1, type=int, help='Current batch for Offline Optimization (-1 for all)')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')

    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0)


    online = args.online

    if not os.path.exists("offline_param_logs"):
        os.makedirs("offline_param_logs", exist_ok=True)
    if not os.path.exists(f"offline_param_logs/{args.ds}"):
        os.makedirs(f"offline_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"offline_param_logs/{args.ds}/{args.online}"):
        os.makedirs(f"offline_param_logs/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists("offline_opt_logs"):
        os.makedirs("offline_opt_logs", exist_ok=True)
    if not os.path.exists(f"offline_opt_logs/{args.ds}"):
        os.makedirs(f"offline_opt_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"offline_opt_logs/{args.ds}/{args.online}"):
        os.makedirs(f"offline_opt_logs/{args.ds}/{args.online}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0,1,2]
    else:
        smac_seeds = [args.smac_seed]

    batch_num = np.ceil(len(data_points) / args.batch)
    if args.batch_id == -1:
        batch_ids = range(batch_num)
    elif args.batch_id >= batch_num:
        raise ValueError(f"Batch id {args.batch_id} is out of range")
    else:
        batch_ids = [args.batch_id]



    name = f"{args.ds}_{args.online}_b{args.batch}_m{args.batch_max}_t{args.batch_budget}"
    args.name = name
    param_log_file = open(f'online_param_logs/{args.ds}/online_params_{name}.csv', 'w', buffering=1)

    i = 0
    for args.batch_id in batch_ids:
        start = args.batch_size*args.batch_id
        stop = min(args.batch_size*(args.batch_id+1), len(data_points))
        batch_labels = labels[start:stop]
        batch_data_points = data_points[start:stop]
        for args.smac_seed in smac_seeds:
            smac = None
            opt_log_file = open(f'online_opt_logs/{args.ds}/online_opt_{name}_o{args.smac_seed}.csv', 'w', buffering=1)

            configspace = get_online_configspace(args.online, batch_data_points, batch_labels, args)
            scenario = Scenario(configspace, use_default_config=True,
                                n_trials=10000000, walltime_limit=args.budget, seed=args.smac_seed, name=name,
                                deterministic=True)
            default_param = configspace.get_default_configuration()
            smac = AlgorithmConfigurationFacade(scenario, online_runner, overwrite=True)
            incumbent = smac.optimize()
            score = smac.runhistory.get_min_cost(incumbent)
            runtime = smac.runhistory.get
            run_num = len(smac.runhistory.items())
            default_score = smac.runhistory.get_min_cost(default_param)
            incumbent_id = smac.runhistory.get_config_id(incumbent)
            opt_log_file.close()
            if i == 0:
                default_prefix = f"online_summaries/{args.ds}/{args.online}/{args.name}_i1"
                param_log_file.write(f"{dict(default_param)};{default_score};-;{default_prefix}\n")
            opt_prefix = f"online_summaries/{args.ds}/{args.online}/{args.name}_i{incumbent_id}"
            param_log_file.write(f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n")
            i += 1
        param_log_file.close()
