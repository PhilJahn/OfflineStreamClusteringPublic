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
from offline_config_handler import get_offline_configspace
from offline_handler import evaluate_offline_clustering
from online_config_handler import get_online_configspace
from online_handler import evaluate_online_clustering
from time_handler import time_limit

global opt_log_file

global batch_data_points
global batch_labels
global smac

non_deterministic_online = ["emc"]
non_deterministic_offline = ["kmeans"]

def online_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    score = np.inf
    print(config_dict)
    online_method = config_dict["online"]
    offline_method = config_dict["offline"]
    metricss = []
    clusteringss = []
    scores = []
    try:
        with time_limit(remaining_time):
            if offline_method in non_deterministic_offline:
                offline_seeds = [0,1,2,3,4]
            else:
                offline_seeds = [0]
            for seed in offline_seeds:
                metrics, clusterings = evaluate_offline_clustering(config_dict["batch_id"], config_dict["offline"], config_dict, seed=seed)
                metricss.append(metrics)
                clusteringss.append(clusterings)
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
            prefix = f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_i{cur_id}"
            for h in range(len(clusteringss)):
                clusterings = clusteringss[h]
                for j in range(len(clusterings)):
                    np.save(f"{prefix}_j{j}_s{h}", clusterings[j])
        opt_log_file.write(f"{config_dict};{score};{prefix}\n")


        for h in range(len(metricss)):
            metric_dict = {"ari": metricss[h]['ari'], "ami": metricss[h]['ami']}
            opt_log_file.write(f"-\t;{h};{metric_dict};{prefix}\n")
        print(f"{config_dict}: {score}", flush=True)


    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Dataset')
    parser.add_argument('--online', default="batchkmeans", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="dbscan", type=str, help='Offline Clustering Method')
    parser.add_argument('--postprocess', default="none", type=str, help='Postprocessing Method')
    parser.add_argument('--batch_budget', default=30
                        , type=int, help='Online SMAC AutoML Budget (in seconds per batch), Offline budget is 2x Online')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--batch_max', default=3, type=int, help='Maximum number of batches')
    parser.add_argument('--batch_id', default=-1, type=int, help='Current batch for Offline Optimization (-1 for all)')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')

    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0)


    online = args.online
    args.postprocess = "none" if args.postprocess is None else args.postprocess

    if not os.path.exists("offline_param_logs"):
        os.makedirs("offline_param_logs", exist_ok=True)
    if not os.path.exists(f"offline_param_logs/{args.ds}"):
        os.makedirs(f"offline_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"offline_param_logs/{args.ds}/{args.online}"):
        os.makedirs(f"offline_param_logs/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"offline_param_logs/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"offline_param_logs/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists("offline_opt_logs"):
        os.makedirs("offline_opt_logs", exist_ok=True)
    if not os.path.exists(f"offline_opt_logs/{args.ds}"):
        os.makedirs(f"offline_opt_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"offline_opt_logs/{args.ds}/{args.online}"):
        os.makedirs(f"offline_opt_logs/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"offline_opt_logs/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"offline_opt_logs/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists("offline_labels"):
        os.makedirs("offline_labels", exist_ok=True)
    if not os.path.exists(f"offline_labels/{args.ds}"):
        os.makedirs(f"offline_labels/{args.ds}", exist_ok=True)
    if not os.path.exists(f"offline_labels/{args.ds}/{args.online}"):
        os.makedirs(f"offline_labels/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"offline_labels/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"offline_labels/{args.ds}/{args.online}/{args.offline}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0,1,2]
    else:
        smac_seeds = [args.smac_seed]

    batch_num = int(np.ceil(len(data_points) / args.batch))
    if args.batch_id == -1:
        batch_ids = range(min(batch_num, args.batch_max))
    elif args.batch_id >= batch_num or args.batch_id >= args.batch_max:
        raise ValueError(f"Batch id {args.batch_id} is out of range")
    else:
        batch_ids = [args.batch_id]

    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_m{args.batch_max}_t{args.batch_budget}_p{args.postprocess}"
    args.name = name
    for args.batch_id in batch_ids:
        i = 0
        if not os.path.exists(f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}"):
            os.makedirs(f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}", exist_ok=True)
        param_log_file = open(f'offline_param_logs/{args.ds}/{args.online}/{args.offline}/offline_params_{name}_c{args.batch_id}.csv', 'w', buffering=1)
        start = args.batch*args.batch_id
        stop = min(args.batch*(args.batch_id+1), len(data_points))
        batch_labels = labels[start:stop]
        batch_data_points = data_points[start:stop]
        for args.smac_seed in smac_seeds:
            smac = None
            opt_log_file = open(f'offline_opt_logs/{args.ds}/{args.online}/{args.offline}/offline_opt_{name}_c{args.batch_id}_o{args.smac_seed}.csv', 'w', buffering=1)

            configspace = get_offline_configspace(args.offline, batch_data_points, batch_labels, args)
            scenario = Scenario(configspace, use_default_config=True,
                                n_trials=10000000, walltime_limit=args.batch_budget, seed=args.smac_seed, name=name,
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
                default_prefix = f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_i1"
                param_log_file.write(f"{dict(default_param)};{default_score};-;{default_prefix}\n")
            opt_prefix = f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_i{incumbent_id}"
            param_log_file.write(f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n")
            i += 1
        param_log_file.close()
