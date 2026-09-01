import argparse
import copy
import math
import os
import traceback

from fastdist import fastdist
from ConfigSpace import Configuration
from pynisher import WallTimeoutException
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from smac import Scenario, AlgorithmConfigurationFacade
from smac import constants as smac_constants

import numpy as np
from torch.utils.data import non_deterministic

import utils
from clustpy.metrics import purity
from datahandler import load_data
from offline_config_handler import get_offline_configspace
from offline_handler import get_summary_clustering
# from denstream import FailedHeuristicError
from optimize_offline import non_deterministic_offline
from original_implementations.birch_sklearn import TooManySummariesError
from smac_variant.algorithm_configuration_facade import NoStoreAlgorithmConfigurationFacade
from subset_methods import NotEnoughDataError
from summary_handler import process_summary
from time_handler import time_limit
from datetime import datetime

from tracker_handler import Tracker

# global opt_log_file

global data_points
global labels
global smac

def pure_offline_runner(config: Configuration, result_tracker: Tracker, seed: int = 0) -> float:
    # print(best_id)

    config_dict = dict(config)
    batch_size = config_dict['batch_size']
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    scores = []
    batch_ids = np.arange(0, len(data_points), step=batch_size).tolist()
    batch_ids.append(len(data_points))
    offline_method = config_dict["offline"]
    timestamps = np.arange(len(data_points))
    # summaries = dict()
    assignments = dict()
    min_stores = dict()
    clusterings = dict()
    summary_clusterings = dict()
    # prior_summaries = dict()
    times = dict()

    radii = np.array([0] * len(data_points))
    weights = np.array([1] * len(data_points))
    statuses = np.array(["dense"] * len(data_points))
    try:
        # main_log_content += f"remaining time: {remaining_time} seconds\n"
        with (time_limit(remaining_time)):
            if offline_method in non_deterministic_offline:
                offline_seeds = list(range(10))
            else:
                offline_seeds = [0]
            for seed in offline_seeds:
                # summaries[seed] = {}
                assignments[seed] = {}
                min_stores[seed] = {}
                clusterings[seed] = {}
                summary_clusterings[seed] = {}
                # prior_summaries[seed] = {}
                times[seed] = {}
                full_times = []
                score = 0


                summary_clustering, full_times = get_summary_clustering(data_points, radii,
                                                                     weights, statuses,
                                                                     offline_method, config_dict, seed)

                score_weight = 0
                for batch in range(len(batch_ids) - 1):
                    # print(batch, flush=True)
                    stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
                    stream_eval_batch = data_points[
                        batch_ids[max(0, batch - config_dict['eval_size'])]:batch_ids[batch + 1]]
                    label_eval_batch = labels[batch_ids[max(0, batch - config_dict['eval_size'])]:batch_ids[batch + 1]]
                    times[seed][batch] = full_times / len(batch_ids)

                    batch_clustering = summary_clustering[
                        batch_ids[max(0, batch - config_dict['eval_size'])]:batch_ids[batch + 1]]

                    offline_ari = adjusted_rand_score(labels_true=label_eval_batch, labels_pred=batch_clustering)
                    offline_ami = adjusted_mutual_info_score(labels_true=label_eval_batch,
                                                             labels_pred=batch_clustering)
                    score += (offline_ari + offline_ami) * len(label_eval_batch)
                    clusterings[seed][batch] = batch_clustering

                    score_weight += len(label_eval_batch)
                scores.append(score / score_weight)
            final_score = 2 - np.mean(scores)  # invert for minimization
    except WallTimeoutException:
        # main_log_content += f"Timed out!\n"
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
        try:
            # print(config_dict, final_score, flush=True)
            incumbent = smac.intensifier.get_incumbent()

            if incumbent is None:
                cur_best_score = np.inf
            else:
                cur_best_score = smac.runhistory.get_min_cost(incumbent)
            if final_score < cur_best_score:
                cur_id = len(smac.runhistory)

                best_store = {}
                best_store["times"] = times
                best_store["clusterings"] = clusterings

                result_tracker.set_best_store(best_store)
                if incumbent is None:
                    result_tracker.set_default_store(best_store)
        except:
            final_score = np.inf
    return final_score, result_tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str,
                        help='Dataset')
    parser.add_argument('--offline', default="kmeans", type=str, help='Offline Clustering Method')
    parser.add_argument('--batch_budget', default="900_logdim"
                        , type=str, help='SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--smac_seed', default=2, type=int, help='Seed for SMAC Optimization')
    parser.add_argument('--summary_limit', default=15000, type=int, help='Limit on Summary structures')
    parser.add_argument('--memory_limit', default=None, type=int, help='Memory Limit in MB')
    parser.add_argument('--trial_limit', default=1000, type=int, help='Maximum number of trials')

    args = parser.parse_args()

    summary_limit = args.summary_limit

    data_points, labels = load_data(args.ds, 0)

    batch_num = int(np.ceil(len(data_points) / args.batch))

    batch_budget_settings = args.batch_budget.split("_")
    batch_base_time = int(batch_budget_settings[0])
    batch_scale = 1
    if len(batch_budget_settings) > 1:
        if batch_budget_settings[1] == "logdim":
            batch_scale = math.log10(len(data_points[0]))
        elif batch_budget_settings[1] == "dim":
            batch_scale = len(data_points[0])

    if args.memory_limit is not None:
        args.memory_limit = int(args.memory_limit) * 1000 * 1000  # MB from B

    args.budget = math.ceil(batch_base_time * batch_scale) * batch_num

    print(data_points.shape, labels.shape,
          f"{args.budget}s, {args.budget / 60}m, {args.budget / 3600}h, {args.budget / (3600 * 24)}d")

    offline = args.offline
    args.online = None
    args.online_batch_budget = args.batch_budget
    args.offline_batch_budget = 0
    args.batch_id = None

    # precompile distance calc
    x_matrix = np.array([[0, 1], [1, 0]])
    x_vector = np.array([[0, 1]])
    dist1 = utils.distance_calc(x_vector, x_vector, "euclidean")
    dist2 = utils.distance_calc(x_matrix, x_vector, "euclidean")
    dist3 = utils.distance_calc(x_matrix, x_matrix, "euclidean")
    if not os.path.exists(f"pure_offline_param_logs/{args.ds}"):
        os.makedirs(f"pure_offline_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"pure_offline_times/{args.ds}/{args.offline}"):
        os.makedirs(f"pure_offline_times/{args.ds}/{args.offline}", exist_ok=True)
    if not os.path.exists(f"pure_offline_labels/{args.ds}/{args.offline}"):
        os.makedirs(f"pure_offline_labels/{args.ds}/{args.offline}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    name = f"{args.ds}_pure_{args.offline}_b{args.batch}_e{args.eval_size}_t{args.batch_budget}_s{args.smac_seed}"
    args.name = name

    name_no_seed = f"{args.ds}_pure_{args.offline}_b{args.batch}_e{args.eval_size}_t{args.batch_budget}"

    param_log_content = ""
    i = 0

    for args.smac_seed in smac_seeds:
        configspace = get_offline_configspace(args.offline, data_points, np.ones(len(data_points)), labels, args)
        if args.memory_limit is not None:
            scenario = Scenario(configspace, use_default_config=True, trial_memory_limit=args.memory_limit,
                                n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed,
                                name=name_no_seed,
                                deterministic=True, termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
        else:
            scenario = Scenario(configspace, use_default_config=True,
                                n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed,
                                name=name_no_seed,
                                deterministic=True, termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
        default_param = configspace.get_default_configuration()
        smac = NoStoreAlgorithmConfigurationFacade(scenario, pure_offline_runner, overwrite=True, logging_level=False)
        incumbent = smac.optimize()
        score = smac.runhistory.get_min_cost(incumbent)
        result_tracker = smac._runner.result_tracker

        run_num = len(smac.runhistory.items())
        default_score = smac.runhistory.get_min_cost(default_param)
        incumbent_id = smac.runhistory.get_config_id(incumbent)

        times_prefix = f"pure_offline_times/{args.ds}/{args.offline}/times_{args.name}"
        clustering_prefix = f"pure_offline_labels/{args.ds}/{args.offline}/clustering_{args.name}"

        best_result_tracker = result_tracker.get_best_store()
        if best_result_tracker is not None:
            # print(incumbent_id, best_id)
            best_times = best_result_tracker["times"]
            best_clustering = best_result_tracker["clusterings"]

            for seed in best_clustering.keys():
                for batch in best_clustering[seed]:
                    np.save(f"{times_prefix}_best_j{batch}_s{seed}", best_times[seed][batch])
                    np.save(f"{clustering_prefix}_best_j{batch}_s{seed}", best_clustering[seed][batch])

        default_result_tracker = result_tracker.get_default_store()
        if default_result_tracker is not None:
            default_times = default_result_tracker["times"]
            default_clustering = default_result_tracker["clusterings"]
            for seed in default_clustering.keys():
                for batch in default_clustering[seed]:
                    np.save(f"{times_prefix}_default_j{batch}_s{seed}", default_times[seed][batch])
                    np.save(f"{clustering_prefix}_default_j{batch}_s{seed}", default_clustering[seed][batch])

            # print(default_result_tracker)

        if i == 0:
            default_assign_prefix = f"pure_offline_labels/{args.ds}/{args.offline}/clustering_{args.name}_default"
            param_log_content += f"{args.smac_seed};{dict(default_param)};{default_score};-;;{default_assign_prefix}\n"
        opt_assign_prefix = f"pure_offline_labels/{args.ds}/{args.offline}/assignment_{args.name}_best"
        param_log_content += f"{args.smac_seed};{dict(incumbent)};{score};{run_num};;{opt_assign_prefix}\n"

        print(smac.intensifier.get_incumbent())
        print(smac.runhistory.get_min_cost(smac.intensifier.get_incumbent()), len(smac.runhistory))

    param_log_file = open(f'pure_offline_param_logs/{args.ds}/pure_offline_params_{name}.csv', 'w', buffering=1)
    param_log_file.write(param_log_content)
    param_log_file.close()
    print("Done")
