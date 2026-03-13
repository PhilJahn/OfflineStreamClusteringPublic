import argparse
import copy
import math
import os
import traceback
from timeit import default_timer

from ConfigSpace import Configuration
from pynisher import WallTimeoutException
from smac import Scenario, AlgorithmConfigurationFacade

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
from summary_handler import process_summary
from time_handler import time_limit
from datetime import datetime

global opt_log_file

global data_points
global labels
global smac

non_deterministic_online = online_handler.non_deterministic_online


def online_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    batch_size = config_dict['batch_size']
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    online_method = config["online"]
    scores = []
    batch_ids = np.arange(0, len(data_points), step=batch_size).tolist()
    batch_ids.append(len(data_points))
    timestamps = np.arange(len(data_points))
    summaries = dict()
    assignments = dict()
    min_stores = dict()
    pseudoclusterings = dict()
    prior_summaries = dict()
    try:
        with time_limit(remaining_time):
            if online_method in non_deterministic_online:
                online_seeds = [0, 1, 2, 3, 4]
            else:
                online_seeds = [0]
            for seed in online_seeds:
                summaries[seed] = {}
                assignments[seed] = {}
                min_stores[seed] = {}
                pseudoclusterings[seed] = {}
                prior_summaries[seed] = {}
                score = 0
                online_clusterer = get_online_method(online_method, config_dict, seed)
                prior_summary = None
                if online_method == "emcstream" or online_method == "emcstream_original" or online_method == "gbfuzzystream_original" or online_method == "birch_sklearn":  # needs to run through fully
                    online_clusterer.learn(data_points.tolist(), timestamps.tolist())
                    full_summary = online_clusterer.get_summary()
                for batch in range(len(batch_ids) - 1):
                    stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
                    label_batch = labels[batch_ids[batch]:batch_ids[batch + 1]]
                    timestamp_batch = timestamps[batch_ids[batch]:batch_ids[batch + 1]]
                    if online_method != "emcstream" and online_method != "emcstream_original" and online_method != "gbfuzzystream_original" and online_method != "birch_sklearn":
                        _, event_log = online_clusterer.learn(stream_batch.tolist(), timestamp_batch.tolist())
                        #print(event_log)
                        summary = online_clusterer.get_summary()
                        if online_method == "clustream":  # subtraction of MCs
                            summary_tmp = copy.deepcopy(summary)
                            removal_ids = []
                            if prior_summary is not None:
                                for mc_old in prior_summary.values():
                                    for mc_new in summary.values():
                                        if mc_old.mc_id in mc_new.mc_history:
                                            mc_new.subtract(mc_old)
                                            if mc_new.count == 0:  # empty MC after subtraction
                                                removal_ids.append(mc_new.mc_id)
                                for del_mc_id in removal_ids:
                                    summary.pop(del_mc_id)

                                centers = np.array([mc.mean for mc in summary.values()])
                                dist = utils.distance_calc(centers, centers, dist="euclidean")
                                mc_ids = list(summary.keys())
                                for mc_index in range(len(mc_ids)):  # update radius for one-MCs
                                    mc_id = mc_ids[mc_index]
                                    mc = summary[mc_id]
                                    if mc.count == 1:
                                        mc.boundary = np.partition(dist[mc_index], 1)[1]
                            prior_summary = summary_tmp
                            prior_summaries[seed][batch] = prior_summary
                    elif online_method == "gbfuzzystream_original" or online_method == "birch_sklearn":
                        summary = full_summary[batch_ids[batch + 1]]
                    else:
                        summary = {"embedding": full_summary["embedding"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "timestamps": full_summary["timestamps"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "labels": full_summary["labels"][batch_ids[batch]:batch_ids[batch + 1]]}
                    summaries[seed][batch] = summary
                    assignment, min_store, psuedoclustering = process_summary(online_method, summary, stream_batch,
                                                                              batch_ids[batch],
                                                                              do_pseudoclustering=True)
                    assignments[seed][batch] = assignment
                    min_stores[seed][batch] = min_store
                    pseudoclusterings[seed][batch] = psuedoclustering
                    #assignpurity = purity(labels_true=label_batch, labels_pred=assignment)
                    pseudopurity = purity(labels_true=label_batch, labels_pred=psuedoclustering)
                    score += pseudopurity * len(label_batch)
                scores.append(score / len(labels))
            final_score = 1 - np.mean(scores)  # invert for minimization
    except WallTimeoutException:
        print("Timed out!", flush=True)
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
        print(traceback.format_exc(), flush=True)
        final_score = np.inf
    # https://stackoverflow.com/questions/9823936/how-do-i-determine-what-type-of-exception-occurred
    except Exception as ex:
        # check against expected exceptions due to hyperparam choice
        processed = False
        if online_method == "denstream" or online_method == "denstream_river":
            if config["mu"] * config["beta"] <= 1: #or type(ex) is FailedHeuristicError:
                final_score = np.inf
                print(traceback.format_exc(), flush=True)
                processed = True
        elif online_method == "dstream" or online_method == "dstream+" or online_method == "dstream++":
            if config["cm"] >= config["dim_grid_num"] ** config["dim"]:
                final_score = np.inf
                print(traceback.format_exc(), flush=True)
                processed = True
        elif online_method == "birch_sklearn":
            if type(ex) is TooManySummariesError:
                final_score = np.inf
                print(traceback.format_exc(), flush=True)
                processed = True
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
            final_score = np.inf
    finally:
        incumbent = smac.intensifier.get_incumbent()
        if incumbent is None:
            cur_best_score = np.inf
        else:
            cur_best_score = smac.runhistory.get_min_cost(incumbent)
        prefix = "-"
        assign_prefix = "-"
        pseudoclustering_prefix = "-"
        mini_prefix = "-"
        if final_score < cur_best_score:
            cur_id = len(smac.runhistory)
            prefix = f"online_summaries/{args.ds}/{args.online}/summary_{args.name}_i{cur_id}"
            assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i{cur_id}"
            pseudoclustering_prefix = f"online_pseudoclusterings/{args.ds}/{args.online}/pseudoclustering_{args.name}_i{cur_id}"
            mini_prefix = f"online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.name}_i{cur_id}"
            prior_prefix = f"online_summaries/{args.ds}/{args.online}/nonsubtracted_summary_{args.name}_i{cur_id}"
            for seed in summaries.keys():
                for batch in summaries[seed]:
                    if online_method != "birch":
                        np.save(f"{prefix}_j{batch}_s{seed}", summaries[seed][batch])
                    np.save(f"{assign_prefix}_j{batch}_s{seed}", assignments[seed][batch])
                    np.save(f"{pseudoclustering_prefix}_j{batch}_s{seed}", pseudoclusterings[seed][batch])
                    np.save(f"{mini_prefix}_j{batch}_s{seed}", min_stores[seed][batch])
                    if online_method == "clustream":
                        np.save(f"{prior_prefix}_j{batch}_s{seed}", prior_summaries[seed][batch])

        opt_log_file.write(
            f"{config_dict};{final_score};{prefix};{assign_prefix};{pseudoclustering_prefix};{mini_prefix}\n")
        for seed_i in range(len(scores)):
            print(np.unique(pseudoclusterings[seed][0]))
            clunum = np.mean([len(np.unique(pseudoclusterings[seed][key])) for key in pseudoclusterings[seed].keys()])
            purity_dict = {"purity": scores[seed_i], "clu_num": clunum}
            opt_log_file.write(f"-\t;{seed_i};{purity_dict};{assign_prefix}\n")
        print(f"{config_dict}: {final_score}", flush=True)
    return final_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str,
                        help='Dataset')
    parser.add_argument('--online', default="gbfuzzystream_original", type=str, help='Online Summarization Method')
    parser.add_argument('--batch_budget', default=1800
                        , type=int, help='SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')

    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0)
    print(data_points.shape, labels.shape)

    batch_num = int(np.ceil(len(data_points) / args.batch))
    args.budget = args.batch_budget * batch_num

    online = args.online
    args.offline = None

    if not os.path.exists("online_param_logs"):
        os.makedirs("online_param_logs", exist_ok=True)
    if not os.path.exists(f"online_param_logs/{args.ds}"):
        os.makedirs(f"online_param_logs/{args.ds}", exist_ok=True)
    if not os.path.exists("online_opt_logs"):
        os.makedirs("online_opt_logs", exist_ok=True)
    if not os.path.exists(f"online_opt_logs/{args.ds}"):
        os.makedirs(f"online_opt_logs/{args.ds}", exist_ok=True)
    if not os.path.exists(f"online_summaries"):
        os.makedirs(f"online_summaries", exist_ok=True)
    if not os.path.exists(f"online_summaries/{args.ds}"):
        os.makedirs(f"online_summaries/{args.ds}", exist_ok=True)
    if not os.path.exists(f"online_summaries/{args.ds}/{args.online}"):
        os.makedirs(f"online_summaries/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_assignments"):
        os.makedirs(f"online_assignments", exist_ok=True)
    if not os.path.exists(f"online_assignments/{args.ds}"):
        os.makedirs(f"online_assignments/{args.ds}", exist_ok=True)
    if not os.path.exists(f"online_assignments/{args.ds}/{args.online}"):
        os.makedirs(f"online_assignments/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_pseudoclusterings"):
        os.makedirs(f"online_pseudoclusterings", exist_ok=True)
    if not os.path.exists(f"online_pseudoclusterings/{args.ds}"):
        os.makedirs(f"online_pseudoclusterings/{args.ds}", exist_ok=True)
    if not os.path.exists(f"online_pseudoclusterings/{args.ds}/{args.online}"):
        os.makedirs(f"online_pseudoclusterings/{args.ds}/{args.online}", exist_ok=True)
    if not os.path.exists(f"online_mini_summaries"):
        os.makedirs(f"online_mini_summaries", exist_ok=True)
    if not os.path.exists(f"online_mini_summaries/{args.ds}"):
        os.makedirs(f"online_mini_summaries/{args.ds}", exist_ok=True)
    if not os.path.exists(f"online_mini_summaries/{args.ds}/{args.online}"):
        os.makedirs(f"online_mini_summaries/{args.ds}/{args.online}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = [0, 1, 2]
    else:
        smac_seeds = [args.smac_seed]

    name = f"{args.ds}_{args.online}_b{args.batch}_t{args.batch_budget}_s{args.smac_seed}"
    args.name = name
    param_log_file = open(f'online_param_logs/{args.ds}/online_params_{name}.csv', 'w', buffering=1)

    i = 0
    for args.smac_seed in smac_seeds:
        opt_log_file = open(f'online_opt_logs/{args.ds}/online_opt_{name}_o{args.smac_seed}.csv', 'w', buffering=1)
        configspace = get_online_configspace(args.online, data_points, labels, args)
        scenario = Scenario(configspace, use_default_config=True,
                            n_trials=10000000, walltime_limit=args.budget, seed=args.smac_seed, name=name,
                            deterministic=True, termination_cost_threshold=[10 ** -18])
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
            default_prefix = f"-1;online_summaries/{args.ds}/{args.online}/summary_{args.name}_i1"
            default_assign_prefix = f"-1;online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i1"
            param_log_file.write(
                f"{args.smac_seed};{dict(default_param)};{default_score};-;{default_prefix};{default_assign_prefix}\n")
        opt_prefix = f"online_summaries/{args.ds}/{args.online}/summary_{args.name}_i{incumbent_id}"
        opt_assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i{incumbent_id}"
        param_log_file.write(
            f"{args.smac_seed};{dict(incumbent)};{score};{run_num};{opt_prefix};{opt_assign_prefix}\n")
        i += 1
    param_log_file.close()

#b: batch
#s: seed
#m: max batches
#o: optimizer seed
#i: incumbent
#j: batch id
#t: batch time budget (for online)
