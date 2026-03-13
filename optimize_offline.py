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

import online_handler
from datahandler import load_data
from offline_config_handler import get_offline_configspace
from offline_handler import get_offline_clustering, get_summary_clustering
from time_handler import time_limit

global opt_log_file

global batch_summary_centers
global batch_summary_weights
global batch_labels
global batch_assignments
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
    summary_clusterings = []
    scores = []
    offline_aris = []
    offline_amis = []
    try:
        with time_limit(remaining_time):
            if offline_method in non_deterministic_offline:
                offline_seeds = [0, 1, 2, 3, 4]
            else:
                offline_seeds = [0]
            for seed in offline_seeds:
                summary_clustering = get_summary_clustering(batch_summary_centers, batch_summary_weights,
                                                            offline_method, config_dict, seed)
                offline_clustering = get_offline_clustering(batch_assignments, summary_clustering)
                offline_ari = adjusted_rand_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_ami = adjusted_mutual_info_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_aris.append(offline_ari)
                offline_amis.append(offline_ami)
                clusterings.append(offline_clustering)
                summary_clusterings.append(summary_clustering)
                scores.append(offline_ari + offline_ami)
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
            prefix_labels = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
            prefix = prefix_labels
            prefix_summary_labels = f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
            for h in range(len(clusterings)):
                np.save(f"{prefix_labels}_s{h}", clusterings[h])
                np.save(f"{prefix_summary_labels}_s{h}", summary_clusterings[h])
        opt_log_file.write(f"{config_dict};{score};{prefix}\n")
        for h in range(len(clusterings)):
            metric_dict = {"ari": offline_aris[h], "ami": offline_amis[h], "score": scores[h]}
            opt_log_file.write(f"-\t;{h};{metric_dict};{prefix}\n")
        print(f"{config_dict}: {score}", flush=True)
    return score


def get_summary_id(args):
    if args.mult:
        prefix = "online_mult"
    else:
        prefix = "online"

    ds = args.ds
    online_alg = args.online
    batch_budget = args.batch_budget
    batch_size = args.batch

    best_i = None
    best_s = None
    best_score = np.inf
    best_optimized = None
    for s in range(3):
        online_param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{s}.csv"
        with open(online_param_path, "r") as online_param_log_file:
            optimized = online_param_log_file.readlines()[-1]
            #print(optimized)
            score = float(optimized.split(";")[2])
            #print(score)
            if score < best_score:
                best_score = score
                best_i = int(optimized.split("i")[-1])
                best_s = s
                best_optimized = optimized
    print(best_s, best_i, best_optimized)

    if args.online in online_handler.non_deterministic_online:  # need to find seed for these
        online_opt_log_path = f"{prefix}_opt_logs/{ds}/{prefix}_opt_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_o{best_s}.csv"
        with open(online_opt_log_path, "r") as online_opt_log_file:
            not_found = True
            cur_line = online_opt_log_file.readline()
            while not_found and cur_line is not None:
                #mini_summary_stream_densired_stmstream_b1000_t1800_s2_i8
                if f"mini_summary_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_i{best_i}" in cur_line:
                    #print(cur_line, flush=True)
                    not_found = False
                cur_line = online_opt_log_file.readline()
            if not not_found:
                non_det_results = []
                while f"assignment_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_i{best_i}" in cur_line:
                    #print(cur_line, flush=True)
                    line_split = cur_line.split(";")
                    #https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
                    result_dict = ast.literal_eval(line_split[2])
                    if args.mult:
                        score = result_dict["score"]
                    else:
                        score = result_dict["purity"]
                    non_det_results.append(score)
                    cur_line = online_opt_log_file.readline()
                #print(non_det_results, flush=True)
                best_nds = np.argmax(non_det_results)
            else:
                raise ValueError
    else:
        best_nds = 0

    return best_s, best_i, best_nds


def get_batch_summary(args, batch_id, best_s, best_i, best_nds):
    if args.mult:
        prefix = "online_mult"
    else:
        prefix = "online"

    ds = args.ds
    online_alg = args.online
    batch_budget = args.batch_budget
    batch_size = args.batch

    # https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
    min_store = np.load(
        f"{prefix}_mini_summaries/{ds}/{online_alg}/mini_summary_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_i{best_i}_j{batch_id}_s{best_nds}.npy",
        allow_pickle=True)
    min_store = min_store.item()
    ids = min_store["id_map"]
    centers = min_store["center"]
    weights = min_store["weights"]
    #radii = min_store["radii"]
    status = min_store["status"]

    non_outlier_indices = [k if min_store["status"][k] in ["dense", "normal"] else "ignore" for k in
                           range(len(min_store["id_map"]))]

    while "ignore" in non_outlier_indices:
        non_outlier_indices.remove("ignore")

    assignments = np.load(
        f"./{prefix}_assignments/{ds}/{online_alg}/assignment_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_i{best_i}_j{batch_id}_s{best_nds}.npy",
        allow_pickle=True)

    non_outlier_assignments = [
        non_outlier_indices.index(assignments[j]) if assignments[j] in non_outlier_indices else -1 for j in
        range(len(assignments))]
    non_outlier_centers = np.array(centers)[non_outlier_indices]
    non_outlier_weights = np.array(weights)[non_outlier_indices]
    return non_outlier_centers, non_outlier_weights, non_outlier_assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--online', default="gbfuzzystream_original", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="dbscan", type=str, help='Offline Clustering Method')
    parser.add_argument('--batch_budget', default=1800
                        , type=int, help='Offline SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--batch_id', default=-1, type=int, help='Current batch for Offline Optimization (-1 for all)')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')
    parser.add_argument('--mult', default=0, type=int, help='Whether to use penalty on unused summary (1) or not (0)')

    args = parser.parse_args()

    args.mult = args.mult == 1

    data_points, labels = load_data(args.ds, 0)

    online = args.online

    if args.mult:
        off_prefix = "offline_mult"
    else:
        off_prefix = "offline"

    if not os.path.exists(f"{off_prefix}_param_logs"):
        os.makedirs(f"{off_prefix}_param_logs", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}/{args.online}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs"):
        os.makedirs(f"{off_prefix}_opt_logs", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs/{args.ds}"):
        os.makedirs(f"{off_prefix}_opt_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs/{args.ds}/{args.online}"):
        os.makedirs(f"{off_prefix}_opt_logs/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_opt_logs/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_opt_logs/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels"):
        os.makedirs(f"{off_prefix}_labels", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}/{args.online}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}/{args.online}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    batch_num = int(np.ceil(len(data_points) / args.batch))
    if args.batch_id == -1:
        batch_ids = range(batch_num)
    elif args.batch_id >= batch_num:
        raise ValueError(f"Batch id {args.batch_id} is out of range")
    else:
        batch_ids = list(range(args.batch_id, min(args.batch_id + 10, batch_num)))

    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_t{args.batch_budget}"
    args.name = name

    best_online_smac_seed, best_incumbent_id, best_nondet_seed = get_summary_id(args)
    args.best_online_smac_seed = best_online_smac_seed
    args.best_incumbent_id = best_incumbent_id
    args.best_nondet_seed = best_nondet_seed

    print(args)

    for args.batch_id in batch_ids:
        if not os.path.exists(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}"):
            os.makedirs(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}", exist_ok=True)
        if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}"):
            os.makedirs(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}", exist_ok=True)
        start = args.batch * args.batch_id
        stop = min(args.batch * (args.batch_id + 1), len(data_points))
        batch_labels = labels[start:stop]
        batch_data_points = data_points[start:stop]
        i = 0
        for args.smac_seed in smac_seeds:
            param_log_file = open(
                f'{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}/{off_prefix}_params_{name}_c{args.batch_id}_o{args.smac_seed}.csv',
                'w', buffering=1)
            smac = None
            opt_log_file = open(
                f'{off_prefix}_opt_logs/{args.ds}/{args.online}/{args.offline}/{off_prefix}_opt_{name}_c{args.batch_id}_o{args.smac_seed}.csv',
                'w', buffering=1)
            batch_summary_centers, batch_summary_weights, batch_assignments = get_batch_summary(args, args.batch_id,
                                                                                                best_online_smac_seed,
                                                                                                best_incumbent_id,
                                                                                                best_nondet_seed)

            if len(batch_summary_centers) > 1:
                configspace = get_offline_configspace(args.offline, batch_summary_centers, batch_summary_weights,
                                                      batch_labels, args)
                print(configspace)
                scenario = Scenario(configspace, use_default_config=True,
                                    n_trials=10000000, walltime_limit=args.batch_budget, seed=args.smac_seed, name=name,
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
                    default_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i1"
                    param_log_file.write(f"{dict(default_param)};{default_score};-;{default_prefix}\n")
                opt_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{incumbent_id}"
                param_log_file.write(f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n")
            else:
                if len(batch_summary_centers) > 0:
                    print("One Summary")
                    offline_clustering = np.ones(len(batch_labels))
                    clusterings = [offline_clustering]
                else:
                    print("Empty Summary")
                    offline_clustering = np.ones(len(batch_labels)) * -1
                    clusterings = [offline_clustering]

                offline_ari = adjusted_rand_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_ami = adjusted_mutual_info_score(labels_true=batch_labels, labels_pred=offline_clustering)
                score = 2 - (offline_ari + offline_ami)

                cur_id = 0
                prefix_labels = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
                prefix = prefix_labels
                prefix_summary_labels = f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
                for h in range(len(clusterings)):
                    np.save(f"{prefix_labels}_s{h}", clusterings[h])
                    np.save(f"{prefix_summary_labels}_s{h}", clusterings[h])
                empty_config = {}
                opt_log_file.write(f"{empty_config};{score};{prefix}\n")
                for h in range(len(clusterings)):
                    metric_dict = {"ari": offline_ari, "ami": offline_ami, "score": 2 - score}
                    opt_log_file.write(f"-\t;{h};{metric_dict};{prefix}\n")
                print(f"{empty_config}: {score}", flush=True)

                default_score = score
                default_param = empty_config
                incumbent_id = 0

                run_num = 0
                incumbent = empty_config

                opt_log_file.close()
                if i == 0:
                    default_prefix = f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i1"
                    param_log_file.write(f"{default_param};{default_score};-;{default_prefix}\n")
                opt_prefix = f"offline_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{incumbent_id}"
                param_log_file.write(f"{incumbent};{score};{run_num};{opt_prefix}\n")

            i += 1
        param_log_file.close()
