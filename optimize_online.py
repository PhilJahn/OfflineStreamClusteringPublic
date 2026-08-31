import argparse
import copy
import math
import os
import traceback

from fastdist import fastdist
from ConfigSpace import Configuration
from pynisher import WallTimeoutException
from smac import Scenario, AlgorithmConfigurationFacade
from smac import constants as smac_constants

import numpy as np
from torch.utils.data import non_deterministic

import online_handler
import utils
from clustpy.metrics import purity
from datahandler import load_data
#from denstream import FailedHeuristicError
from online_config_handler import get_online_configspace
from online_handler import get_online_method
from original_implementations.birch_sklearn import TooManySummariesError
from smac_variant.algorithm_configuration_facade import NoStoreAlgorithmConfigurationFacade
from subset_methods import NotEnoughDataError
from summary_handler import process_summary
from time_handler import time_limit
from datetime import datetime

from tracker_handler import Tracker

#global opt_log_file

global data_points
global labels
global smac

global summary_limit

non_deterministic_online = online_handler.non_deterministic_online

class SummaryLimitError(ValueError):
    pass


def online_runner(config: Configuration, result_tracker: Tracker, seed: int = 0) -> float:
    #print(best_id)

    config_dict = dict(config)
    batch_size = config_dict['batch_size']
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    #print(config_dict, flush=True)
    #print("remaining time:", remaining_time, "seconds", flush=True)
    online_method = config["online"]
    scores = []
    batch_ids = np.arange(0, len(data_points), step=batch_size).tolist()
    batch_ids.append(len(data_points))
    timestamps = np.arange(len(data_points))
    #summaries = dict()
    assignments = dict()
    min_stores = dict()
    all_pseudoclusterings = dict()
    #prior_summaries = dict()
    times = dict()
    try:
        #main_log_content += f"remaining time: {remaining_time} seconds\n"
        with (time_limit(remaining_time)):
            if online_method in non_deterministic_online:
                online_seeds = [0] # one fixed seed for non-det also-> prioritize HPs over seed stability
            else:
                online_seeds = [0]
            for seed in online_seeds:
                #summaries[seed] = {}
                assignments[seed] = {}
                min_stores[seed] = {}
                #prior_summaries[seed] = {}
                times[seed] = {}
                all_pseudoclusterings[seed] = {}
                full_times = []
                score = 0
                online_clusterer = get_online_method(online_method, config_dict, seed)
                prior_summary = None
                if online_method == "emcstream" or online_method == "emcstream_original" or online_method == "emcstream_mod": # needs to run through fully
                    _, _, full_mems, full_times = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
                    full_summary = online_clusterer.get_summary()
                elif online_method == "gbfuzzystream_original" or online_method == "birch_sklearn":  # needs to run through fully, but does have batches internally
                    _, _, full_mems, full_times = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
                    full_summary, full_summary_mems = online_clusterer.get_summary()
                score_weight = 0
                for batch in range(len(batch_ids) - 1):
                    #print(batch, flush=True)
                    stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
                    stream_eval_batch = data_points[batch_ids[max(0, batch-config_dict['eval_size'])]:batch_ids[batch + 1]]
                    label_eval_batch = labels[batch_ids[max(0, batch-config_dict['eval_size'])]:batch_ids[batch + 1]]
                    timestamp_batch = timestamps[batch_ids[batch]:batch_ids[batch + 1]]
                    if (online_method != "emcstream" and online_method != "emcstream_original" and online_method != "emcstream_mod"
                        and online_method != "gbfuzzystream_original" and online_method != "birch_sklearn"):
                        _, event_log, batch_mems, batch_times = online_clusterer.learn(stream_batch.tolist(), timestamp_batch.tolist())
                        #print(event_log)
                        summary = online_clusterer.get_summary()
                        times[seed][batch] = np.sum(batch_times)
                        #print(mems[seed][batch], times[seed][batch], summary_mems[seed][batch])
                    elif online_method == "gbfuzzystream_original" or online_method == "birch_sklearn":
                        summary = full_summary[batch_ids[batch + 1]]
                        times[seed][batch] = full_times[batch_ids[batch + 1]]
                        #print(mems[seed][batch], times[seed][batch], summary_mems[seed][batch])
                    else:
                        summary = {"embedding": full_summary["embedding"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "timestamps": full_summary["timestamps"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "labels": full_summary["labels"][batch_ids[batch]:batch_ids[batch + 1]]}
                        times[seed][batch] = full_times/len(batch_ids) # take avg
                        all_pseudoclusterings[seed][batch] = summary["labels"]
                    #summaries[seed][batch] = summary
                    if online_method == "emcstream" or online_method == "emcstream_original" or online_method == "emcstream_mod":
                        summary_count = len(summary["labels"])
                    elif online_method == "denstream" or online_method == "denstream_river":
                        summary_count = len(summary["pmcs"])
                    else:
                        summary_count = len(summary)

                    if summary_count > summary_limit:
                        raise SummaryLimitError

                    #print(len(summary))
                    #print(mems[seed][batch], summary_mems[seed][batch], flush=True)
                    assignment, min_store = process_summary(online_method, summary, stream_batch, stream_eval_batch,
                                                                              batch_ids[batch],
                                                                              do_pseudoclustering=False)
                    min_store = copy.deepcopy(min_store)

                    #print(min_store)
                    assignments[seed][batch] = assignment
                    min_stores[seed][batch] = min_store
                    #pseudoclusterings[seed][batch] = psuedoclustering
                    if online_method in ["emcstream", "emcstream_original", "emcstream_mod"]:
                        pseudoclustering = np.array(summary["labels"])[assignment]
                        pseudopurity = purity(labels_true=label_eval_batch, labels_pred=pseudoclustering)
                        score += pseudopurity * len(label_eval_batch)
                    else:
                        noise_filter = [k if min_store["status"][k] in ["dense", "normal"] else -1 for k in
                                               range(len(min_store["id_map"]))]
                        noise_filter.append(-1)
                        noise_filter = np.array(noise_filter)
                        #print(np.array(assignment))
                        #print(noise_filter[np.array(assignment)])

                        #print(np.unique(noise_filter[np.array(assignment)]), len(np.unique(noise_filter[np.array(assignment)])), np.unique(np.array(assignment)))
                        assignpurity = purity(labels_true=label_eval_batch, labels_pred=noise_filter[np.array(assignment)])
                    #pseudopurity = purity(labels_true=label_eval_batch, labels_pred=psuedoclustering)
                        score += assignpurity * len(label_eval_batch)
                    score_weight += len(label_eval_batch)
                scores.append(score / score_weight)
                del online_clusterer
            final_score = 1 - np.mean(scores)  # invert for minimization
    except WallTimeoutException:
        #main_log_content += f"Timed out!\n"
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
        #print(traceback.format_exc(), flush=True)
        final_score = np.inf
        raise AssertionError
    except SummaryLimitError:
        #main_log_content += f"{traceback.format_exc()}\n"
        #main_log_content += f"Too Many Summaries: {len(summary)}\n"
        final_score = np.inf
        raise SummaryLimitError(len(summary))
        #print(traceback.format_exc(), flush=True)
    # https://stackoverflow.com/questions/9823936/how-do-i-determine-what-type-of-exception-occurred
    except Exception as ex:
        # check against expected exceptions due to hyperparam choice
        processed = False
        #print(traceback.format_exc(), flush=True)
        if online_method == "denstream" or online_method == "denstream_river":
            if config["mu"] * config["beta"] <= 1: #or type(ex) is FailedHeuristicError:
                final_score = np.inf
                #print(traceback.format_exc(), flush=True)
                processed = True
        elif online_method == "dstream" or online_method == "dstream+" or online_method == "dstream++":
            if config["cm"] >= config["dim_grid_num"] ** config["dim"]:
                final_score = np.inf
                #print(traceback.format_exc(), flush=True)
                processed = True
        elif online_method == "birch_sklearn":
            if type(ex) is TooManySummariesError:
                final_score = np.inf
                #print(traceback.format_exc(), flush=True)
                processed = True
        elif "window" in online_method:
            if type(ex) is NotEnoughDataError:
                final_score = np.inf
                #print(traceback.format_exc(), flush=True)
                processed = True
        #if not processed:
            #if not os.path.exists("./error_logs/"):
            #    os.makedirs("./error_logs/", exist_ok=True)
            #timestr = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")

            #error_log_content += f"{timestr}\n"
            #error_log_content += f"{str(config_dict)}\n"
            #error_file = open(f'./error_logs/error_{type(ex).__name__}_{name}_{timestr}.txt', 'w', buffering=1)
            #error_file.write(str(config_dict))
            #error_file.write("\n")
            #error_file.write(traceback.format_exc())
            #error_file.close()
            #print(traceback.format_exc(), flush=True)
            #error_log_content += f"{traceback.format_exc()}\n"
            #error_log_content += "\n"
            #main_log_content += f"{traceback.format_exc()}\n"
        final_score = np.inf

        raise ex
    finally:
        try:
            #print(config_dict, final_score, flush=True)
            incumbent = smac.intensifier.get_incumbent()

            if incumbent is None:
                cur_best_score = np.inf
            else:
                cur_best_score = smac.runhistory.get_min_cost(incumbent)
            if final_score < cur_best_score:
                cur_id = len(smac.runhistory)


                best_store = {}
                best_store["id"] = cur_id
                best_store["assignments"] = assignments
                best_store["min_stores"] = min_stores
                best_store["times"] = times
                if online_method == "emcstream" or online_method == "emcstream_original":
                    best_store["pseudoclusterings"] = all_pseudoclusterings
                    
                result_tracker.set_best_store(best_store)                    
                if incumbent is None:
                    result_tracker.set_default_store(best_store)

            # opt_log_content += f"{config_dict};{final_score};{is_incumbent}\n"
            # for seed_i in range(len(scores)):
            #     #print(np.unique(pseudoclusterings[seed][0]))
            #     clunum = np.mean([len(np.unique(pseudoclusterings[seed_i][key])) for key in pseudoclusterings[seed_i].keys()])
            #     result_dict = {"score": float(scores[seed_i]), "clu_num": float(clunum), "mem": float(max(mems[seed_i].values())), "summary_mem": float(max(summary_mems[seed_i].values())), "time": float(sum(times[seed_i].values()))}
            #     opt_log_content += f"-\t;{seed_i};{result_dict};{is_incumbent}\n"
            #main_log_content += f"{config_dict}: {final_score}\n"
        except:
            final_score = np.inf
            #main_log_content += f"{traceback.format_exc()}\n"
            #print(traceback.format_exc(), flush=True)
    return final_score, result_tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str,
                        help='Dataset')
    parser.add_argument('--online', default="full_batch", type=str, help='Online Summarization Method')
    parser.add_argument('--batch_budget', default="1_logdim"
                        , type=str, help='SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--smac_seed', default=0, type=int, help='Seed for SMAC Optimization')
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
        args.memory_limit = int(args.memory_limit) * 1000 * 1000 # MB from B


    args.budget = math.ceil(batch_base_time * batch_scale) * batch_num

    print(data_points.shape, labels.shape,f"{args.budget}s, {args.budget / 60}m, {args.budget / 3600}h, {args.budget / (3600 * 24)}d")

    online = args.online
    args.offline = None

    #precompile distance calc
    x_matrix = np.array([[0,1], [1,0]])
    x_vector = np.array([[0,1]])
    dist1 = utils.distance_calc(x_vector, x_vector, "euclidean")
    dist2 = utils.distance_calc(x_matrix, x_vector, "euclidean")
    dist3 = utils.distance_calc(x_matrix, x_matrix, "euclidean")
    if not os.path.exists(f"online_param_logs/{args.ds}"):
        os.makedirs(f"online_param_logs/{args.ds}", exist_ok=True)
    # if not os.path.exists(f"online_opt_logs/{args.ds}"):
    #     os.makedirs(f"online_opt_logs/{args.ds}", exist_ok=True)
    #if not os.path.exists(f"online_summaries/{args.ds}/{args.online}"):
    #    os.makedirs(f"online_summaries/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_assignments/{args.ds}/{args.online}"):
        os.makedirs(f"online_assignments/{args.ds}/{args.online}", exist_ok=True)
    #if not os.path.exists(f"online_pseudoclusterings/{args.ds}/{args.online}"):
    #    os.makedirs(f"online_pseudoclusterings/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_mini_summaries/{args.ds}/{args.online}"):
        os.makedirs(f"online_mini_summaries/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_times/{args.ds}/{args.online}"):
        os.makedirs(f"online_times/{args.ds}/{args.online}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    name = f"{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget}_s{args.smac_seed}"
    args.name = name

    name_no_seed = f"{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget}"

    param_log_content = ""
    i = 0

    for args.smac_seed in smac_seeds:
        configspace = get_online_configspace(args.online, data_points, labels, args)
        if args.memory_limit is not None:
            scenario = Scenario(configspace, use_default_config=True, trial_memory_limit=args.memory_limit,
                                n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed, name=name_no_seed,
                                deterministic=True, termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
        else:
            scenario = Scenario(configspace, use_default_config=True,
                                n_trials=args.trial_limit, walltime_limit=args.budget, seed=args.smac_seed, name=name_no_seed,
                                deterministic=True, termination_cost_threshold=[smac_constants.VERY_SMALL_NUMBER])
        default_param = configspace.get_default_configuration()
        smac = NoStoreAlgorithmConfigurationFacade(scenario, online_runner, overwrite=True, logging_level=False)
        incumbent = smac.optimize()
        score = smac.runhistory.get_min_cost(incumbent)
        result_tracker = smac._runner.result_tracker

        run_num = len(smac.runhistory.items())
        default_score = smac.runhistory.get_min_cost(default_param)
        incumbent_id = smac.runhistory.get_config_id(incumbent)
        # opt_log_file = open(f'online_opt_logs/{args.ds}/online_opt_{name}_o{args.smac_seed}.csv', 'w', buffering=1)
        # opt_log_file.write(opt_log_content)
        # opt_log_file.close()

        #prefix = f"online_summaries/{args.ds}/{args.online}/summary_{args.name}"
        assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}"
        pseudoclustering_prefix = f"online_assignments/{args.ds}/{args.online}/pseudoclustering_{args.name}"

        #pseudoclustering_prefix = f"online_pseudoclusterings/{args.ds}/{args.online}/pseudoclustering_{args.name}"
        mini_prefix = f"online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.name}"
        #prior_prefix = f"online_summaries/{args.ds}/{args.online}/nonsubtracted_summary_{args.name}"
        times_prefix = f"online_times/{args.ds}/{args.online}/times_{args.name}"
        
        best_result_tracker = result_tracker.get_best_store()
        if best_result_tracker is not None:
            best_min_stores = best_result_tracker["min_stores"]
            best_id = best_result_tracker["id"]
            #print(incumbent_id, best_id)
            best_times = best_result_tracker["times"]
            best_assignments = best_result_tracker["assignments"]
            if args.online in ["emcstream", "emcstream_original"]:
                best_pseudoclusterings = best_result_tracker["pseudoclusterings"]

            for seed in best_min_stores.keys():
                for batch in best_min_stores[seed]:
                    #if args.online != "birch":
                    #    np.save(f"{prefix}_i{best_id}_j{batch}_s{seed}", best_summaries[seed][batch])
                    np.save(f"{assign_prefix}_best_j{batch}_s{seed}", best_assignments[seed][batch])
                    #np.save(f"{pseudoclustering_prefix}_i{best_id}_j{batch}_s{seed}", best_pseudoclusterings[seed][batch])
                    np.save(f"{mini_prefix}_best_j{batch}_s{seed}", best_min_stores[seed][batch])
                    np.save(f"{times_prefix}_best_j{batch}_s{seed}", best_times[seed][batch])

                    if args.online in ["emcstream", "emcstream_original"]:
                        np.save(f"{pseudoclustering_prefix}_best_j{batch}_s{seed}", best_pseudoclusterings[seed][batch])
                    #if args.online == "clustream":
                    #    np.save(f"{prior_prefix}_i{best_id}_j{batch}_s{seed}", best_prior_summaries[seed][batch])

            #print(best_result_tracker)

        
        default_result_tracker = result_tracker.get_default_store()
        if default_result_tracker is not None:
            default_min_stores = default_result_tracker["min_stores"]
            default_id = default_result_tracker["id"]
            default_times = default_result_tracker["times"]
            default_assignments = default_result_tracker["assignments"]
            if args.online in ["emcstream", "emcstream_original"]:
                default_pseudoclusterings = default_result_tracker["pseudoclusterings"]
            for seed in default_min_stores.keys():
                for batch in default_min_stores[seed]:
                    #if args.online != "birch":
                    #    np.save(f"{prefix}_i0_j{batch}_s{seed}", default_summaries[seed][batch])
                    np.save(f"{assign_prefix}_default_j{batch}_s{seed}", default_assignments[seed][batch])
                    #np.save(f"{pseudoclustering_prefix}_i0_j{batch}_s{seed}", default_pseudoclusterings[seed][batch])
                    np.save(f"{mini_prefix}_default_j{batch}_s{seed}", default_min_stores[seed][batch])
                    np.save(f"{times_prefix}_default_j{batch}_s{seed}", default_times[seed][batch])

                    if args.online in ["emcstream", "emcstream_original"]:
                        np.save(f"{pseudoclustering_prefix}_default_j{batch}_s{seed}", default_pseudoclusterings[seed][batch])
                    #if args.online == "clustream":
                    #    np.save(f"{prior_prefix}_i0_j{batch}_s{seed}", default_prior_summaries[seed][batch])

            #print(default_result_tracker)


        if i == 0:
            default_prefix = f"online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.name}_default"
            default_assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_default"
            param_log_content += f"{args.smac_seed};{dict(default_param)};{default_score};-;{default_prefix};{default_assign_prefix}\n"
        opt_prefix = f"online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.name}_best"
        opt_assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_best"
        param_log_content += f"{args.smac_seed};{dict(incumbent)};{score};{run_num};{opt_prefix};{opt_assign_prefix}\n"


        print(smac.intensifier.get_incumbent())
        print(smac.runhistory.get_min_cost(smac.intensifier.get_incumbent()), len(smac.runhistory))


    param_log_file = open(f'online_param_logs/{args.ds}/online_params_{name}.csv', 'w', buffering=1)
    param_log_file.write(param_log_content)
    param_log_file.close()
    print("Done")

#b: batch
#s: seed
#m: max batches
#o: optimizer seed
#i: incumbent
#j: batch id
#t: batch time budget (for online)
