import argparse
import math
import os
import traceback
from pprint import pprint

import numpy as np
from sklearn.metrics import calinski_harabasz_score

from bop_score import calculate_bop
from datahandler import load_data
from metrics.disco import disco_score
from result_evaluator import simplified_silhouette_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--online', default="denstream_river", type=str, help='Online Summarization Method')
    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--batch_budget_online', default="300_logdim"
                        , type=str, help='Online SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    args = parser.parse_args()

    store_name = f"summary_evals/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}.npy"

    if not os.path.exists("summary_evals"):
        os.makedirs("summary_evals", exist_ok=True)


    data_points, labels = load_data(args.ds)

    batch_ids = math.ceil(len(labels)/args.batch)

    print(batch_ids, flush=True)

    all_metrics = {}
    all_metrics_std = {}

    best_score = 2
    for s in range(3):
        online_param_path = f"online_param_logs/{args.ds}/online_params_{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget_online}_s{s}.csv"
        with open(online_param_path, "r") as online_param_log_file:
            optimized = online_param_log_file.readlines()[-1]
            score = float(optimized.split(";")[2])
            if score < best_score:
                best_score = score
                best_s = s
                best_optimized = optimized

    all_metrics = {}
    for batch_id in range(batch_ids):
        all_metrics[batch_id] = {}
        start = args.batch * batch_id
        stop = min(args.batch * (batch_id + 1), len(data_points))
        batch_labels = labels[max(0, start - (args.eval_size*args.batch)):stop]
        batch_data_points = data_points[max(0, start - (args.eval_size*args.batch)):stop]

        min_store = np.load(
            f"online_mini_summaries/{args.ds}/{args.online}/mini_summary_{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget_online}_s{best_s}_best_j{batch_id}_s{0}.npy",
            allow_pickle=True)
        min_store = min_store.item()

        assignments = np.load(
            f"./online_assignments/{args.ds}/{args.online}/assignment_{args.ds}_{args.online}_b{args.batch}_e{args.eval_size}_t{args.batch_budget_online}_s{best_s}_best_j{batch_id}_s{0}.npy",
            allow_pickle=True)

        label_ctr = {}
        for j in range(len(min_store["center"])):
            label_ctr[j] = {}
        for k in range(len(assignments)):
            if batch_labels[k] not in label_ctr[assignments[k]]:
                label_ctr[assignments[k]][batch_labels[k]] = 1
            else:
                label_ctr[assignments[k]][batch_labels[k]] += 1

        weight_labels = []
        weight_centers = []
        weight_labels_100 = []
        weight_centers_100 = []
        cur_noise_label = -1
        for i in range(len(min_store["center"])):
            cur_max = 0
            cur_label = -1
            label_count = 0
            label_options = []
            for label_option in label_ctr[i].keys():
                label_count += label_ctr[i][label_option]
                label_options.append(label_option)
                if cur_max < label_ctr[i][label_option]:
                    cur_max = label_ctr[i][label_option]
                    cur_label = label_option

            label_weights = []

            if len(label_options) == 0:
                label_options.append(cur_noise_label)
                cur_noise_label -= 1
                label_weights.append(1)
            else:
                for label_option in label_ctr[i].keys():
                    label_weight = label_ctr[i][label_option] / label_count
                    label_weights.append(label_weight)

            weight = min_store["weights"][i]
            dupli_weight = int(np.round(weight))
            # dupli_weight_100 = int(np.round(weight) * 100)

            label_option_counts = dupli_weight * np.array(label_weights)

            remaining_nums = dupli_weight
            while remaining_nums > 0:
                remaining_nums -= 1
                label_choice = np.argmax(label_option_counts)
                weight_labels.append(label_options[label_choice])
                label_option_counts[label_choice] -= 1
                weight_centers.append(min_store["center"][i])

        noise_filter = [k if min_store["status"][k] in ["dense", "normal"] else -1 for k in
                        range(len(min_store["id_map"]))]
        noise_filter.append(-1)
        nonoise_filter = np.where(np.array(noise_filter) > -1)
        # print(noise_filter)
        # print(nonoise_filter)

        centers = min_store["center"][nonoise_filter]
        weights = min_store["weights"][nonoise_filter]

        try:
            bop_score_real = calculate_bop(batch_data_points, np.ones(len(batch_data_points)), centers, weights, 100)
            bop_score_summary = calculate_bop(centers, weights, batch_data_points, np.ones(len(batch_data_points)), 100)
        except:
            bop_score_real = 1
            bop_score_summary = 1
        all_metrics[batch_id]["bop_score_real"] = bop_score_real
        all_metrics[batch_id]["bop_score_summary"] = bop_score_summary
        all_metrics[batch_id]["bop_score"] = bop_score_summary + bop_score_real

        print(np.array(weight_centers).shape)

            # label_option_counts_100 = dupli_weight_100 * np.array(label_weights)
            # remaining_nums_100 = dupli_weight_100
            # while remaining_nums_100 > 0:
            #     remaining_nums_100 -= 1
            #     label_choice_100 = np.argmax(label_option_counts_100)
            #     weight_labels_100.append(label_options[label_choice_100])
            #     label_option_counts_100[label_choice_100] -= 1
            #     weight_centers_100.append(min_store["center"][i])

            # print(label_options, weight_labels)
        try:
            batch_ssil, _ , _ = simplified_silhouette_score(np.array(weight_centers), np.array(weight_labels))
        #    batch_ssil_100, _, _ = simplified_silhouette_score(np.array(weight_centers_100), np.array(weight_labels_100))
        except:
            batch_ssil = -1
            # batch_ssil_100 = -1
        if math.isnan(batch_ssil):
            batch_ssil = -1
        all_metrics[batch_id]["summary_ssil"] = batch_ssil
        # all_metrics[batch_id]["summary_ssil_100"] = batch_ssil_100

        try:
            batch_chi = calinski_harabasz_score(np.array(weight_centers), np.array(weight_labels))
        except:
            batch_chi = 0
        if math.isnan(batch_chi):
            batch_chi = 0

        all_metrics[batch_id]["summary_chi"] = batch_chi
        all_metrics[batch_id]["weight"] = len(batch_labels)

    all_metrics["all"] = {}
    for metric in all_metrics[0].keys():
        metric_sum = 0
        metric_std_sum = 0
        weight_sum = 0
        for batch_id in range(batch_ids):
            metric_sum += all_metrics[batch_id][metric] * all_metrics[batch_id]["weight"]
            weight_sum += all_metrics[batch_id]["weight"]
        all_metrics["all"][metric] = metric_sum / weight_sum

    pprint(all_metrics)
    np.save(store_name, all_metrics)