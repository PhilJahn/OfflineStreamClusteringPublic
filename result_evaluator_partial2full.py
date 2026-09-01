import argparse
import math
import os
from pprint import pprint

import numpy as np

from datahandler import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="sensorstream", type=str, help='Dataset')
    parser.add_argument('--online', default="emcstream_original", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="online", type=str, help='Offline Clustering Method')
    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--batch_budget_online', default="300_logdim"
                        , type=str, help='Online SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch_budget_offline', default="600_logdim"
                        , type=str, help='Offline SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--overwrite', default=1, type=int, help='Overwrite existing results')
    args = parser.parse_args()

    args.overwrite = args.overwrite == 1

    if not os.path.exists("partial_result_evals"):
        os.makedirs("partial_result_evals", exist_ok=True)

    if not os.path.exists(f"partial_result_evals/{args.offline}"):
        os.makedirs(f"partial_result_evals/{args.offline}", exist_ok=True)

    if args.offline == "online":
        store_name_final = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}.npy"
        store_name_std_final = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}_std.npy"

    else:
        store_name_final = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}.npy"
        store_name_std_final = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}_std.npy"


    if os.path.exists(store_name_final) and not args.overwrite:
        raise FileExistsError(store_name_final)


    data_points, labels = load_data(args.ds)

    batch_ids = math.ceil(len(labels)/args.batch)

    print(batch_ids, flush=True)

    all_metrics = {}
    all_metrics_std = {}

    full_labels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    min_clu = np.max(labels) +1
    full_min_clu = {0: min_clu, 1: min_clu, 2: min_clu, 3: min_clu, 4: min_clu, 5: min_clu, 6: min_clu, 7: min_clu, 8: min_clu, 9: min_clu}


    for batch_id in np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])*100:
        try:

            if args.offline == "online":
                store_name = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}_{batch_id}-{batch_id+100}.npy"
                store_name_std = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}_{batch_id}-{batch_id+100}_std.npy"
            else:
                store_name = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}_{batch_id}-{batch_id+100}.npy"
                store_name_std = f"partial_result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}_{batch_id}-{batch_id+100}_std.npy"
            batch_scores = np.load(store_name, allow_pickle=True).item()
            batch_stds = np.load(store_name_std, allow_pickle=True).item()


            for k in range(batch_id, batch_id+100):
                all_metrics[k] = batch_scores[k]
                all_metrics_std[k] = batch_stds[k]
        except:
            print(f"end at {batch_id}")
            break

    print(all_metrics)
    all_metrics["all"] = {}
    all_metrics["all_std"] = {}

    print(all_metrics)
    print(all_metrics_std)
    for metric in all_metrics[0].keys():
        metric_sum = 0
        metric_std_sum = 0
        weight_sum = 0
        for batch_id in range(batch_ids):
            metric_sum += all_metrics[batch_id][metric] * all_metrics[batch_id]["weight"]
            metric_std_sum += all_metrics_std[batch_id][metric] * all_metrics[batch_id]["weight"]
            weight_sum += all_metrics[batch_id]["weight"]
        all_metrics["all"][metric] = metric_sum / weight_sum
        all_metrics["all_std"][metric] = metric_std_sum / weight_sum

    pprint(all_metrics)

    np.save(store_name_final, all_metrics)
    np.save(store_name_std_final, all_metrics_std)