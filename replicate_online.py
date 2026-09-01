import argparse
import ast
import copy
import os
import traceback

import numpy as np

from clustpy.metrics import purity
from datahandler import load_data
from online_handler import get_online_method
from optimize_online import SummaryLimitError
from summary_handler import process_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="sensorstream", type=str,
                        help='Dataset')
    parser.add_argument('--online', default="emcstream_original", type=str, help='Online Summarization Method')
    parser.add_argument('--batch_budget', default="300_logdim"
                        , type=str, help='SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--summary_limit', default=15000, type=int, help='Limit on Summary structures')
    parser.add_argument('--memory_limit', default=None, type=int, help='Memory Limit in MB')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random state')
    args = parser.parse_args()

    data_points, labels = load_data(args.ds, 0)

    best_s = None
    best_score = np.inf
    best_optimized = None
    ds = args.ds
    online_alg = args.online
    online_batch_budget = args.batch_budget
    batch_size = args.batch
    eval_size = args.eval_size

    seed = args.seed


    for s in range(3):
        online_param_path = f"online_param_logs/{ds}/online_params_{ds}_{online_alg}_b{batch_size}_e{eval_size}_t{online_batch_budget}_s{s}.csv"
        with open(online_param_path, "r") as online_param_log_file:
            optimized = online_param_log_file.readlines()[-1]
            #print(optimized)
            score = float(optimized.split(";")[2])
            #print(score)
            if score < best_score:
                best_score = score
                best_s = s
                best_hps = ast.literal_eval(optimized.split(";")[1])
                best_optimized = optimized
        print(score)

    name = f"{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget}_s{best_s}"
    args.name = name
    try:
        assignments = {}
        min_stores = {}
        times = {}
        all_pseudoclusterings = {}
        full_times = []
        batch_ids = np.arange(0, len(data_points), step=batch_size).tolist()
        batch_ids.append(len(data_points))
        timestamps = np.arange(len(data_points))
        score = 0
        online_clusterer = get_online_method(online_alg, best_hps, seed)
        prior_summary = None
        if online_alg == "emcstream" or online_alg == "emcstream_original" or online_alg == "emcstream_mod":  # needs to run through fully
            _, _, full_mems, full_times = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
            full_summary = online_clusterer.get_summary()
        elif online_alg == "gbfuzzystream_original" or online_alg == "birch_sklearn":  # needs to run through fully, but does have batches internally
            _, _, full_mems, full_times = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
            full_summary, full_summary_mems = online_clusterer.get_summary()
        score_weight = 0
        for batch in range(len(batch_ids) - 1):
            # print(batch, flush=True)
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            stream_eval_batch = data_points[
                batch_ids[max(0, batch - args.eval_size)]:batch_ids[batch + 1]]
            label_eval_batch = labels[batch_ids[max(0, batch - args.eval_size)]:batch_ids[batch + 1]]
            timestamp_batch = timestamps[batch_ids[batch]:batch_ids[batch + 1]]
            if (online_alg != "emcstream" and online_alg != "emcstream_original" and online_alg != "emcstream_mod"
                    and online_alg != "gbfuzzystream_original" and online_alg != "birch_sklearn"):
                _, event_log, batch_mems, batch_times = online_clusterer.learn(stream_batch.tolist(),
                                                                               timestamp_batch.tolist())
                # print(event_log)
                summary = online_clusterer.get_summary()
                times[batch] = np.sum(batch_times)
                # print(mems[batch], times[batch], summary_mems[batch])
            elif online_alg == "gbfuzzystream_original" or online_alg == "birch_sklearn":
                summary = full_summary[batch_ids[batch + 1]]
                times[batch] = full_times[batch_ids[batch + 1]]
                # print(mems[batch], times[batch], summary_mems[batch])
            else:
                summary = {"embedding": full_summary["embedding"][batch_ids[batch]:batch_ids[batch + 1]],
                           "timestamps": full_summary["timestamps"][batch_ids[batch]:batch_ids[batch + 1]],
                           "labels": full_summary["labels"][batch_ids[batch]:batch_ids[batch + 1]]}
                times[batch] = full_times / len(batch_ids)  # take avg
                all_pseudoclusterings[batch] = summary["labels"]
            # summaries[batch] = summary
            if online_alg == "emcstream" or online_alg == "emcstream_original" or online_alg == "emcstream_mod":
                summary_count = len(summary["labels"])
            elif online_alg == "denstream" or online_alg == "denstream_river":
                summary_count = len(summary["pmcs"])
            else:
                summary_count = len(summary)

            if summary_count > args.summary_limit:
                raise SummaryLimitError

            # print(len(summary))
            # print(mems[batch], summary_mems[batch], flush=True)
            assignment, min_store = process_summary(online_alg, summary, stream_batch, stream_eval_batch,
                                                    batch_ids[batch],
                                                    do_pseudoclustering=False)
            min_store = copy.deepcopy(min_store)

            # print(min_store)
            assignments[batch] = assignment
            min_stores[batch] = min_store
            # pseudoclusterings[batch] = psuedoclustering
            if online_alg in ["emcstream", "emcstream_original", "emcstream_mod"]:
                pseudoclustering = np.array(summary["labels"])[assignment]
                pseudopurity = purity(labels_true=label_eval_batch, labels_pred=pseudoclustering)
                score += pseudopurity * len(label_eval_batch)
            else:
                noise_filter = [k if min_store["status"][k] in ["dense", "normal"] else -1 for k in
                                range(len(min_store["id_map"]))]
                noise_filter.append(-1)
                noise_filter = np.array(noise_filter)
                assignpurity = purity(labels_true=label_eval_batch, labels_pred=noise_filter[np.array(assignment)])
                score += assignpurity * len(label_eval_batch)
            score_weight += len(label_eval_batch)
        final_score = 1 - (score / score_weight)
    except:
        print(traceback.format_exc(), flush=True)
    finally:
        if not os.path.exists(f"replicate_online_assignments/{args.ds}/{args.online}"):
            os.makedirs(f"replicate_online_assignments/{args.ds}/{args.online}", exist_ok=True)
        if not os.path.exists(f"replicate_online_mini_summaries/{args.ds}/{args.online}"):
            os.makedirs(f"replicate_online_mini_summaries/{args.ds}/{args.online}", exist_ok=True)
        if not os.path.exists(f"replicate_online_times/{args.ds}/{args.online}"):
            os.makedirs(f"replicate_online_times/{args.ds}/{args.online}", exist_ok=True)

        assign_prefix = f"replicate_online_assignments/{args.ds}/{args.online}/assignment_{args.name}"
        pseudoclustering_prefix = f"replicate_online_assignments/{args.ds}/{args.online}/pseudoclustering_{args.name}"
        mini_prefix = f"replicate_online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.name}"
        times_prefix = f"replicate_online_times/{args.ds}/{args.online}/times_{args.name}"

        for batch in min_stores:
            np.save(f"{assign_prefix}_best_j{batch}_s{seed}", assignments[batch])
            np.save(f"{mini_prefix}_best_j{batch}_s{seed}", min_stores[batch])
            np.save(f"{times_prefix}_best_j{batch}_s{seed}", times[batch])
            #if args.online in ["emcstream", "emcstream_original"]:
            #    np.save(f"{pseudoclustering_prefix}_best_j{batch}_s{seed}", pseudoclustering[batch])

        print(final_score)
        print("done")


