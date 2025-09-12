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
from online_config_handler import get_online_configspace
from online_handler import evaluate_online_clustering
from time_handler import time_limit

global opt_log_file

global data_points
global labels
global smac

non_deterministic_online = ["emc", "ikmeans", "bkmeans"]

# batch
# ikmeans
# w ikmeans
# bkmeans
# w bkmeans
# clustream
# w clustream
# denstream
# w denstream
# emc
# streamkm
# gbfuzzy
# msmst
# dstream
# clustree
# w clustree
# PCMC
# w PCMC


def online_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    score = np.inf
    online_method = config["online"]
    metricss = []
    summariess = []
    assignmentss = []
    scores = []
    try:
        with time_limit(remaining_time):
            if online_method in non_deterministic_online:
                online_seeds = [0,1,2,3,4]
            else:
                online_seeds = [0]
            for seed in online_seeds:
                metrics, assignments, summaries = evaluate_online_clustering(data_points, labels, online_method, config_dict, config["batch_size"], return_summary=True, eval_mode="purity", seed=seed)
                summariess.append(summaries)
                assignmentss.append(assignments)
                metricss.append(metrics)
                scores.append(1-metrics["purity"])
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
        assign_prefix = "-"
        if score < cur_best_score:
            cur_id = len(smac.runhistory)
            prefix = f"online_summaries/{args.ds}/{args.online}/summary_{args.name}_i{cur_id}"
            assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i{cur_id}"


            for h in range(len(summariess)):
                summaries = summariess[h]
                assignments = assignmentss[h]
                summary_num = len(summaries)
                assert summary_num == np.ceil(len(data_points) / config["batch_size"])
                for j in range(summary_num):
                    summary = summaries[j]
                    np.save(f"{prefix}_j{j}_s{h}", summary)
                    assignment = assignments[j]
                    np.save(f"{assign_prefix}_j{j}_s{h}", assignment)

        opt_log_file.write(f"{config_dict};{score};{prefix};{assign_prefix}\n")

        for h in range(len(metricss)):
            purity_dict = {"purity": metricss[h]['purity']}
            opt_log_file.write(f"-\t;{h};{purity_dict};\n")
        print(f"{config_dict}: {score}", flush=True)


    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="densired2", type=str, help='Dataset')
    parser.add_argument('--online', default="clustream", type=str, help='Online Summarization Method')
    parser.add_argument('--batch_budget', default=30
                        , type=int, help='SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--batch_max', default=3, type=int, help='Maximum number of batches')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')

    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0, args.batch, args.batch_max)
    print(data_points.shape, labels.shape)

    batch_num = np.ceil(len(data_points) / args.batch)
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

    if args.smac_seed == -1:
        smac_seeds = [0,1,2]
    else:
        smac_seeds = [args.smac_seed]

    name = f"{args.ds}_{args.online}_b{args.batch}_m{args.batch_max}_t{args.batch_budget}"
    args.name = name
    param_log_file = open(f'online_param_logs/{args.ds}/online_params_{name}.csv', 'w', buffering=1)

    i = 0
    for args.smac_seed in smac_seeds:
        smac = None
        opt_log_file = open(f'online_opt_logs/{args.ds}/online_opt_{name}_o{args.smac_seed}.csv', 'w', buffering=1)

        configspace = get_online_configspace(args.online, data_points, labels, args)
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
            default_prefix = f"-1;online_summaries/{args.ds}/{args.online}/summary_{args.name}_i1"
            default_assign_prefix = f"-1;online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i1"
            param_log_file.write(f"{args.smac_seed};{dict(default_param)};{default_score};-;{default_prefix};{default_assign_prefix}\n")
        opt_prefix = f"online_summaries/{args.ds}/{args.online}/summary_{args.name}_i{incumbent_id}"
        opt_assign_prefix = f"online_assignments/{args.ds}/{args.online}/assignment_{args.name}_i{incumbent_id}"
        param_log_file.write(f"{args.smac_seed};{dict(incumbent)};{score};{run_num};{opt_prefix};{opt_assign_prefix}\n")
        i+=1
    param_log_file.close()

#b: batch
#s: seed
#m: max batches
#o: optimizer seed
#i: incumbent
#j: batch id
#t: batch time budget (for online)