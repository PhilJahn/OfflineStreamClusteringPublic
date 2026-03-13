import argparse
import ast
from datetime import datetime
import math
import os
import traceback
from timeit import default_timer

from ConfigSpace import Configuration
from pynisher import WallTimeoutException
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from smac import Scenario, AlgorithmConfigurationFacade

import numpy as np
from torch.utils.data import non_deterministic

from datahandler import load_data
from offline_config_handler import get_offline_configspace
from offline_handler import get_offline_clustering, get_summary_clustering
from time_handler import time_limit

global opt_log_file

global data_points
global labels
global smac
global off_prefix

non_deterministic_offline = ["kmeans"]


def offline_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    score = np.inf
    #print(config_dict)
    offline_method = config_dict["offline"]
    clusterings = []
    scores = []
    offline_aris = []
    offline_amis = []

    batch_size = config_dict['batch_size']
    batch_ids = np.arange(0, len(data_points), step=batch_size).tolist()
    batch_ids.append(len(data_points))
    try:
        with time_limit(remaining_time):
            if offline_method in non_deterministic_offline:
                offline_seeds = [0, 1, 2, 3, 4]
            else:
                offline_seeds = [0]
            scores = []
            for seed in offline_seeds:
                offline_clustering = get_summary_clustering(data_points, [1]*len(data_points),
                                                            offline_method, config_dict, seed)
                clusterings.append(offline_clustering)
                offline_score = 0
                for batch in range(len(batch_ids) - 1):
                    label_batch = labels[batch_ids[batch]:batch_ids[batch + 1]]
                    cluster_batch = offline_clustering[batch_ids[batch]:batch_ids[batch + 1]]
                    ari_batch = adjusted_rand_score(labels_true=label_batch, labels_pred=cluster_batch)
                    ami_batch = adjusted_mutual_info_score(labels_true=label_batch, labels_pred=cluster_batch)
                    offline_aris.append(ari_batch)
                    offline_amis.append(ami_batch)
                    offline_score += (ari_batch + ami_batch) * len(label_batch)
                scores.append(offline_score / len(labels))
            score = 2 - np.mean(scores)

    except WallTimeoutException:
        print("Timed out!", flush=True)
        score = np.inf
        raise WallTimeoutException("Timed out!")
    except AssertionError:
        if not os.path.exists("./assertion_error_logs/"):
            os.makedirs("./assertion_error_logs/", exist_ok=True)
        timestr = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
        assertion_error_file = open(f'./assertion_error_logs/assert_{name}_{timestr}.txt', 'w', buffering=1)
        assertion_error_file.write(str(config_dict))
        assertion_error_file.write("\n")
        assertion_error_file.write(traceback.format_exc())
        assertion_error_file.close()
        print(traceback.format_exc(), flush=True)
        score = np.inf
    # https://stackoverflow.com/questions/9823936/how-do-i-determine-what-type-of-exception-occurred
    except Exception as ex:
        # check against expected exceptions due to hyperparam choice
        processed = False
        if not processed:
            if not os.path.exists("./error_logs/"):
                os.makedirs("./error_logs/", exist_ok=True)
            timestr = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
            error_file = open(f'./error_logs/error_{type(ex).__name__}_{name}_{timestr}.txt', 'w', buffering=1)
            error_file.write(str(config_dict))
            error_file.write("\n")
            error_file.write(traceback.format_exc())
            error_file.close()
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
            prefix_labels = f"{off_prefix}_labels/{args.ds}/{args.offline}/{args.name}_call_o{args.smac_seed}_i{cur_id}"
            prefix = prefix_labels
            prefix_summary_labels = f"{off_prefix}_summary_labels/{args.ds}/{args.offline}/{args.name}_call_o{args.smac_seed}_i{cur_id}"
            for h in range(len(clusterings)):
                np.save(f"{prefix_labels}_s{h}", clusterings[h])
                np.save(f"{prefix_summary_labels}_s{h}", clusterings[h])
        opt_log_file.write(f"{config_dict};{score};{prefix}\n")
        for h in range(len(clusterings)):
            metric_dict = {"ari": offline_aris[h], "ami": offline_amis[h], "score": scores[h]}
            opt_log_file.write(f"-\t;{h};{metric_dict};{prefix}\n")
        print(f"{config_dict}: {score}", flush=True)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--offline', default="kmeans", type=str, help='Offline Clustering Method')
    parser.add_argument('--batch_budget', default=3600
                        , type=int, help='Offline SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')

    args = parser.parse_args()
    args.online = "pure_offline"

    data_points, labels = load_data(args.ds, 0)

    batch_num = int(np.ceil(len(data_points) / args.batch))
    args.budget = args.batch_budget * batch_num

    online = args.online
    off_prefix = "pure_offline"

    if not os.path.exists(f"{off_prefix}_param_logs"):
        os.makedirs(f"{off_prefix}_param_logs", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}/{args.offline}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs"):
        os.makedirs(f"{off_prefix}_opt_logs", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs/{args.ds}"):
        os.makedirs(f"{off_prefix}_opt_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs/{args.ds}/{args.offline}"):
        os.makedirs(f"{off_prefix}_opt_logs/{args.ds}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels"):
        os.makedirs(f"{off_prefix}_labels", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}/{args.offline}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}/{args.offline}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}/{args.offline}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_t{args.batch_budget}"
    args.name = name

    args.batch_id = "all"

    print(args)

    i = 0
    for args.smac_seed in smac_seeds:
        param_log_file = open(
            f'{off_prefix}_param_logs/{args.ds}/{args.offline}/{off_prefix}_params_{name}_call_o{args.smac_seed}.csv',
            'w', buffering=1)
        smac = None
        opt_log_file = open(
            f'{off_prefix}_opt_logs/{args.ds}/{args.offline}/{off_prefix}_opt_{name}_call_o{args.smac_seed}.csv',
            'w', buffering=1)

        configspace = get_offline_configspace(args.offline, data_points, np.array([1]*len(data_points)), labels, args)
        print(configspace)
        scenario = Scenario(configspace, use_default_config=True,
                            n_trials=10000000, walltime_limit=args.budget, seed=args.smac_seed, name=name,
                            deterministic=True, termination_cost_threshold=[10**-18])
        default_param = configspace.get_default_configuration()
        smac = AlgorithmConfigurationFacade(scenario, offline_runner, overwrite=True)
        incumbent = smac.optimize()
        score = smac.runhistory.get_min_cost(incumbent)
        runtime = smac.runhistory.get
        run_num = len(smac.runhistory.items())
        default_score = smac.runhistory.get_min_cost(default_param)
        incumbent_id = smac.runhistory.get_config_id(incumbent)

        opt_log_file.close()
        if i == 0:
            default_prefix = f"{off_prefix}_labels/{args.ds}/{args.offline}/{args.name}_call_o{args.smac_seed}_i1"
            param_log_file.write(f"{dict(default_param)};{default_score};-;{default_prefix}\n")
        opt_prefix = f"{off_prefix}_labels/{args.ds}/{args.offline}/{args.name}_call_o{args.smac_seed}_i{incumbent_id}"
        param_log_file.write(f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n")
        i += 1
    param_log_file.close()
