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
from smac import Scenario
from smac import constants as smac_constants
from smac_variant.algorithm_configuration_facade import NoStoreAlgorithmConfigurationFacade
import numpy as np
from torch.utils.data import non_deterministic

import online_handler
from datahandler import load_data
from offline_config_handler import get_offline_configspace
from offline_handler import get_offline_clustering, get_summary_clustering
from time_handler import time_limit
from tracker_handler import Tracker

global batch_summary_centers
global batch_summary_radii
global batch_summary_weights
global batch_summary_status
global batch_labels
global batch_assignments
global smac
global off_prefix


non_deterministic_offline = ["kmeans"]


def offline_runner(config: Configuration, result_tracker: Tracker, seed: int = 0) -> float:
    config_dict = dict(config)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    #print("remaining time:", remaining_time, "seconds")
    final_score = np.inf
    #print(config_dict)
    offline_method = config_dict["offline"]
    clusterings = []
    summary_clusterings = []
    times = []
    scores = []
    offline_aris = []
    offline_amis = []
    try:
        with time_limit(remaining_time):
            if offline_method in non_deterministic_offline:
                offline_seeds = list(range(10))
            else:
                offline_seeds = [0]
            for seed in offline_seeds:
                summary_clustering, runtime = get_summary_clustering(batch_summary_centers, batch_summary_radii,
                                                                     batch_summary_weights, batch_summary_status,
                                                                     offline_method, config_dict, seed)
                offline_clustering = get_offline_clustering(batch_assignments, summary_clustering)
                offline_ari = adjusted_rand_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_ami = adjusted_mutual_info_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_aris.append(offline_ari)
                offline_amis.append(offline_ami)
                clusterings.append(offline_clustering)
                times.append(runtime)
                summary_clusterings.append(summary_clustering)
                scores.append(offline_ari + offline_ami)
            final_score = 2 - np.mean(scores)
    except WallTimeoutException:
        final_score = np.inf
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
        # print(traceback.format_exc(), flush=True)
        final_score = np.inf
        raise AssertionError
    # https://stackoverflow.com/questions/9823936/how-do-i-determine-what-type-of-exception-occurred
    except Exception as ex:
        # check against expected exceptions due to hyperparam choice
        processed = False
        final_score = np.inf
        raise ex
    finally:
        incumbent = smac.intensifier.get_incumbent()
        if incumbent is None:
            cur_best_score = np.inf
        else:
            cur_best_score = smac.runhistory.get_min_cost(incumbent)



        if final_score < cur_best_score:
            cur_id = len(smac.runhistory)

            best_store = {}
            best_store["id"] = cur_id
            best_store["clusterings"] = clusterings
            best_store["summary_clusterings"] = summary_clusterings
            best_store["times"] = times

            result_tracker.set_best_store(best_store)
            if incumbent is None:
                result_tracker.set_default_store(best_store)

    #     cur_id = len(smac.runhistory)
        #     prefix_labels = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
        #     prefix = prefix_labels
        #     prefix_summary_labels = f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_i{cur_id}"
        #     for h in range(len(clusterings)):
        #         np.save(f"{prefix_labels}_s{h}", clusterings[h])
        #         np.save(f"{prefix_summary_labels}_s{h}", summary_clusterings[h])
        # #print(f"{config_dict};{score};{prefix}")
        # for h in range(len(clusterings)):
        #     metric_dict = {"ari": offline_aris[h], "ami": offline_amis[h], "score": scores[h]}
        #     #print(f"-\t;{h};{metric_dict};{prefix}")
        # print(f"{config_dict}: {score}", flush=True)
    return final_score, result_tracker


def get_summary_id(args):
    prefix = "online"

    ds = args.ds
    online_alg = args.online
    online_batch_budget = args.online_batch_budget
    batch_size = args.batch
    eval_size = args.eval_size
    best_s = None
    best_score = np.inf
    best_optimized = None
    for s in range(3):
        online_param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_{online_alg}_b{batch_size}_e{eval_size}_t{online_batch_budget}_s{s}.csv"
        with open(online_param_path, "r") as online_param_log_file:
            optimized = online_param_log_file.readlines()[-1]
            #print(optimized)
            score = float(optimized.split(";")[2])
            #print(score)
            if score < best_score:
                best_score = score
                best_s = s
                best_optimized = optimized
    print(best_s, best_optimized)

    # if args.online in online_handler.non_deterministic_online:  # need to find seed for these
    #     online_opt_log_path = f"{prefix}_opt_logs/{ds}/{prefix}_opt_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_o{best_s}.csv"
    #     with open(online_opt_log_path, "r") as online_opt_log_file:
    #         not_found = True
    #         cur_line = online_opt_log_file.readline()
    #         while not_found and cur_line is not None:
    #             #mini_summary_stream_densired_stmstream_b1000_t1800_s2_i8
    #             if f"mini_summary_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_best" in cur_line:
    #                 #print(cur_line, flush=True)
    #                 not_found = False
    #             cur_line = online_opt_log_file.readline()
            # if not not_found:
            #     non_det_results = []
            #     while f"assignment_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_best" in cur_line:
            #         #print(cur_line, flush=True)
            #         line_split = cur_line.split(";")
            #         #https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
            #         result_dict = ast.literal_eval(line_split[2])
            #         if args.mult:
            #             score = result_dict["score"]
            #         else:
            #             score = result_dict["purity"]
            #         non_det_results.append(score)
            #         cur_line = online_opt_log_file.readline()
            #     #print(non_det_results, flush=True)
            #     best_nds = np.argmax(non_det_results)
            # else:
            #     raise ValueError
    # else:
    best_nds = 0

    return best_s, best_nds


def get_batch_summary(args, batch_id, best_s, best_nds):
    prefix = "online"
    ds = args.ds
    online_alg = args.online
    online_batch_budget = args.online_batch_budget
    batch_size = args.batch
    eval_size = args.eval_size

    # https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
    min_store = np.load(
        f"{prefix}_mini_summaries/{ds}/{online_alg}/mini_summary_{ds}_{online_alg}_b{batch_size}_e{eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch_id}_s{best_nds}.npy",
        allow_pickle=True)
    min_store = min_store.item()
    ids = min_store["id_map"]
    centers = min_store["center"]
    weights = min_store["weights"]
    radii = min_store["radii"]
    status = min_store["status"]

    #print(np.unique(status))


    if online_alg not in ["emcstream", "emcstream_original"]:
        non_outlier_indices = [k if min_store["status"][k] in ["dense", "normal"] else "ignore" for k in
                               range(len(min_store["id_map"]))]
    else:
        non_outlier_indices = list(range(len(np.array(centers))))

    while "ignore" in non_outlier_indices:
        non_outlier_indices.remove("ignore")

    assignments = np.load(
        f"./{prefix}_assignments/{ds}/{online_alg}/assignment_{ds}_{online_alg}_b{batch_size}_e{eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch_id}_s{best_nds}.npy",
        allow_pickle=True)

    non_outlier_assignments = [
        non_outlier_indices.index(assignments[j]) if assignments[j] in non_outlier_indices else -1 for j in
        range(len(assignments))]
    non_outlier_centers = np.array(centers)[non_outlier_indices]
    non_outlier_radii = np.array(radii)[non_outlier_indices]
    non_outlier_weights = np.array(weights)[non_outlier_indices]
    non_outlier_status = np.array(status)[non_outlier_indices]

    return non_outlier_centers, non_outlier_radii, non_outlier_weights, non_outlier_status, non_outlier_assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--online', default="dstream", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="adjacency", type=str, help='Offline Clustering Method')
    parser.add_argument('--online_batch_budget', default="1_logdim"
                        , type=str, help='Online SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--offline_batch_budget', default="1_logdim"
                        , type=str, help='Offline SMAC AutoML Budget (in seconds per batch)')

    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--batch_id', default=0, type=int, help='Current batch for Offline Optimization (-1 for all)')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')
    parser.add_argument('--memory_limit', default=None, type=int, help='Memory Limit in MB')
    parser.add_argument('--trial_limit', default=1000, type=int, help='Maximum number of trials')


    args = parser.parse_args()

    if args.offline == "dbscan" and args.online in ["dstream", "denstream_river"]: # different DBSCAN handling for non-integer weights
        args.offline = "fdbscan"

    data_points, labels = load_data(args.ds, 0)

    online = args.online

    off_prefix = "offline"

    if not os.path.exists(f"{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"{off_prefix}_times/{args.ds}/{args.online}/{args.offline}"):
        os.makedirs(f"{off_prefix}_times/{args.ds}/{args.online}/{args.offline}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    batch_num = int(np.ceil(len(data_points) / args.batch))
    # if args.batch_id == -1:
    #     batch_ids = range(batch_num)
    # elif args.batch_id >= batch_num:
    #     raise ValueError(f"Batch id {args.batch_id} is out of range")
    # else:
    #     batch_ids = list(range(args.batch_id, min(args.batch_id + 10, batch_num)))

    batch_ids = [args.batch_id]

    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_t1{args.online_batch_budget}_t2{args.offline_batch_budget}"
    args.name = name

    best_online_smac_seed, best_nondet_seed = get_summary_id(args)
    args.best_online_smac_seed = best_online_smac_seed
    args.best_nondet_seed = best_nondet_seed

    batch_budget_settings = args.offline_batch_budget.split("_")
    batch_base_time = int(batch_budget_settings[0])
    batch_scale = 1
    if len(batch_budget_settings) > 1:
        if batch_budget_settings[1] == "logdim":
            batch_scale = math.log10(len(data_points[0]))
        elif batch_budget_settings[1] == "dim":
            batch_scale = len(data_points[0])
    args.budget = math.ceil(batch_base_time * batch_scale)

    if args.memory_limit is not None:
        args.memory_limit = int(args.memory_limit) * 1000 * 1000 # MB from B


    print(args)

    print(data_points.shape, labels.shape,f"{args.budget}s, {args.budget / 60}m, {args.budget / 3600}h, {args.budget / (3600 * 24)}d (x {batch_num})")

    for args.batch_id in batch_ids:
        label_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}"
        summary_label_prefix = f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}"
        times_prefix = f"{off_prefix}_times/{args.ds}/{args.online}/{args.offline}/"

        batch_name = name + f"/{args.batch_id}"

        if not os.path.exists(label_prefix):
            os.makedirs(label_prefix, exist_ok=True)
        if not os.path.exists(summary_label_prefix):
            os.makedirs(summary_label_prefix, exist_ok=True)
        if not os.path.exists(times_prefix):
            os.makedirs(times_prefix, exist_ok=True)
        start = args.batch * args.batch_id
        stop = min(args.batch * (args.batch_id + 1), len(data_points))
        batch_labels = labels[max(0, start - (args.eval_size*args.batch)):stop]
        i = 0
        param_log_content = ""
        for args.smac_seed in smac_seeds:
            smac = None
            batch_summary_centers, batch_summary_radii, batch_summary_weights, batch_summary_status, batch_assignments = (
                get_batch_summary(args, args.batch_id, best_online_smac_seed, best_nondet_seed))

            print(len(batch_summary_centers))

            if len(batch_summary_centers) > 1:
                configspace = get_offline_configspace(args.offline, batch_summary_centers, batch_summary_weights,
                                                      batch_labels, args)
                print(configspace)
                if args.memory_limit is not None:
                    scenario = Scenario(configspace, use_default_config=True, trial_memory_limit=args.memory_limit,
                                        n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed, name="offline/"+batch_name,
                                        deterministic=True,
                                        termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
                else:
                    scenario = Scenario(configspace, use_default_config=True,
                                        n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed, name="offline/"+batch_name,
                                        deterministic=True,
                                        termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
                default_param = configspace.get_default_configuration()
                smac = NoStoreAlgorithmConfigurationFacade(scenario, offline_runner, overwrite=True, logging_level=False)
                incumbent = smac.optimize()
                score = smac.runhistory.get_min_cost(incumbent)
                result_tracker = smac._runner.result_tracker
                run_num = len(smac.runhistory.items())
                default_score = smac.runhistory.get_min_cost(default_param)
                incumbent_id = smac.runhistory.get_config_id(incumbent)

                
                best_result_tracker = result_tracker.get_best_store()
                if best_result_tracker is not None:
                    offline_best_clustering = best_result_tracker["clusterings"]
                    offline_best_id = best_result_tracker["id"]
                    offline_best_times = best_result_tracker["times"]
                    offline_best_summary_clustering = best_result_tracker["summary_clusterings"]
                else:
                    offline_best_times = None
                    offline_best_summary_clustering = None
                    offline_best_id = None
                    offline_best_clustering = None
                    
                default_result_tracker = result_tracker.get_default_store()
                if default_result_tracker is not None:
                    offline_default_clustering = default_result_tracker["clusterings"]
                    offline_default_id = default_result_tracker["id"]
                    offline_default_times = default_result_tracker["times"]
                    offline_default_summary_clustering = default_result_tracker["summary_clusterings"]
                else:
                    offline_default_times = None
                    offline_default_summary_clustering = None
                    offline_default_id = None
                    offline_default_clustering = None


                if i == 0:
                    default_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.name}_c{args.batch_id}_o{args.smac_seed}_default"
                    param_log_content += f"{dict(default_param)};{default_score};-;{default_prefix}\n"
                opt_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.name}_c{args.batch_id}_o{args.smac_seed}_best"
                param_log_content += f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n"

                print(smac.intensifier.get_incumbent())
                print(smac.runhistory.get_min_cost(smac.intensifier.get_incumbent()), len(smac.runhistory))
                del smac._runner
                del smac
            else:
                if len(batch_summary_centers) > 0:
                    #print("One Summary")
                    offline_clustering = np.ones(len(batch_labels))
                    summary_clustering = [1]

                else:
                    #print("Empty Summary")
                    offline_clustering = np.ones(len(batch_labels)) * -1
                    summary_clustering = []

                offline_ari = adjusted_rand_score(labels_true=batch_labels, labels_pred=offline_clustering)
                offline_ami = adjusted_mutual_info_score(labels_true=batch_labels, labels_pred=offline_clustering)
                score = 2 - (offline_ari + offline_ami)

                cur_id = 0
                empty_config = {}
                default_score = score
                default_param = empty_config
                incumbent_id = 0

                run_num = 0
                incumbent = empty_config
                clustering_list = []
                summary_clustering_list = []
                time_list = []
                if args.offline in non_deterministic_offline:
                    seeds = list(range(10))
                else:
                    seeds = [0]

                for seed in seeds:
                    clustering_list.append(offline_clustering)
                    summary_clustering_list.append(summary_clustering)
                    time_list.append(0)


                offline_default_clustering = clustering_list
                offline_default_summary_clustering = summary_clustering_list
                offline_default_times = time_list
                offline_best_clustering = clustering_list
                offline_best_summary_clustering = summary_clustering_list
                offline_best_times = time_list
                offline_best_id = 0

                if i == 0:
                    default_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_default"
                    param_log_content += f"{dict(default_param)};{score};-;{default_prefix}\n"
                opt_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}/{args.batch_id}/{args.name}_c{args.batch_id}_o{args.smac_seed}_best"
                param_log_content += f"{dict(incumbent)};{score};{run_num};{opt_prefix}\n"

            #print(offline_best_summary_clustering)
            #print(offline_best_clustering)
            if offline_default_clustering is not None:  # if default produced usable results
                for seed in range(len(offline_default_clustering)):
                    np.save(f"{label_prefix}/clustering_{args.name}_o{args.smac_seed}_default_j{args.batch_id}_s{seed}", offline_default_clustering[seed])
                    np.save(f"{summary_label_prefix}/summary_clustering_{args.name}_o{args.smac_seed}_default_j{args.batch_id}_s{seed}", offline_default_summary_clustering[seed])
                    np.save(f"{times_prefix}/times_{args.name}_o{args.smac_seed}_default_j{args.batch_id}_s{seed}", offline_default_times[seed])

            if offline_best_clustering is not None:  # if default produced usable results
                for seed in range(len(offline_best_clustering)):
                    np.save(f"{label_prefix}/clustering_{args.name}_o{args.smac_seed}_best_j{args.batch_id}_s{seed}", offline_best_clustering[seed])
                    np.save(f"{summary_label_prefix}/summary_clustering_{args.name}_o{args.smac_seed}_best_j{args.batch_id}_s{seed}", offline_best_summary_clustering[seed])
                    np.save(f"{times_prefix}/times_{args.name}_o{args.smac_seed}_best_j{args.batch_id}_s{seed}", offline_best_times[seed])

            i += 1
        param_log_file = open(
            f'{off_prefix}_param_logs/{args.ds}/{args.online}/{args.offline}/{off_prefix}_params_{name}_c{args.batch_id}_o{args.smac_seed}.csv',
            'w', buffering=1)
        param_log_file.write(param_log_content)
        param_log_file.close()
    print("done", flush=True)