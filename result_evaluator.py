import argparse
import ast
import math
import os
import warnings
from pprint import pprint

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    fowlkes_mallows_score, homogeneity_score, normalized_mutual_info_score, completeness_score

from clustpy.metrics import purity, unsupervised_clustering_accuracy, \
    variation_of_information, \
    fair_normalized_mutual_information
from datahandler import load_data
from metrics.disco import disco_score
from online_handler import non_deterministic_online
from temporalsilhouette.TSindex import tempsil
from utils import distance_calc

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)

# simplified silhouette score (Wang, Fei, et al. "An analysis of the application of simplified silhouette to the evaluation of k-means clustering validity." International Conference on Machine Learning and Data Mining in Pattern Recognition. Cham: Springer International Publishing, 2017.)
def simplified_silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    a = np.zeros(len(labels))
    b = np.ones(len(labels)) * np.inf
    for label in unique_labels:
        cluster_index = np.where(labels == label)
        not_cluster_index = np.where(labels != label)
        cluster = X[cluster_index]
        cluster_mean = np.mean(cluster, axis=0)

        dists = distance_calc(X, cluster_mean, dist="euclidean")
        a[cluster_index] = dists[cluster_index]

        b[not_cluster_index] = np.minimum(dists[not_cluster_index], b[not_cluster_index])

    element_scores = (b - a) / np.maximum(b, a)

    return np.mean(element_scores), np.mean(a), np.mean(b)


def get_clustering(args):

    online = args.online
    offline = args.offline
    online_batch_budget = args.batch_budget_online
    offline_batch_budget = args.batch_budget_offline

    if offline == "online":
        prefix = "online"
        label_folder = "online_assignments"
        label_name = "assignment"
    elif offline == "seed_online":
        prefix = "online"
        label_folder = "online_assignments"
        label_name = "assignment"
    elif offline == "pure_online":
        prefix = "pure_online"
        label_folder = "pure_online_labels"
        label_name = ""
    elif online == "pure_offline":
        prefix = "pure_offline"
        label_folder = "pure_offline_labels"
        label_name = ""
    else:
        prefix = "offline"
        label_folder = "offline_labels"
        label_name = ""

    if online == "dstream" or online == "denstream_river":
        if offline == "dbscan":
            offline = "fdbscan"

    ds = args.ds
    online_alg = online
    offline_alg = offline
    batch_size = args.batch

    clusterings = {}


    if prefix == "offline":
        for batch_id in range(batch_ids):
            best_s = None
            best_score = np.inf
            best_optimized = None
            clusterings[batch_id] = {}
            for s in range(3):
                try:
                    param_path = f"{prefix}_param_logs/{ds}/{online_alg}/{offline_alg}/{prefix}_params_{ds}_{online_alg}_{offline_alg}_b{batch_size}_t1{online_batch_budget}_t2{offline_batch_budget}_c{batch_id}_o{s}.csv"
                    with open(param_path, "r") as param_log_file:
                        content = param_log_file.readlines()
                        if len(content) > 0:
                            optimized = content[-1]
                            print(optimized)
                            score = float(optimized.split(";")[1])
                            print(score)
                            if score < best_score:
                                best_score = score
                                best_s = s
                                best_optimized = optimized
                except:
                    print(f"{s} yielded no results")
            if best_optimized is None:
                raise Exception("No results")
            print(batch_id, best_s, best_score, best_optimized.replace("\n", ""), flush=True)
            if offline_alg == "kmeans":
                seeds = list(range(10))
            else:
                seeds = [0]

            for seed_index in range(len(seeds)):
                print("loaded seed", seeds[seed_index])
                clustering = np.load(
                    f"{label_folder}/{ds}/{online_alg}/{offline_alg}/clustering_{ds}_{online_alg}_{offline_alg}_b{batch_size}_t1{online_batch_budget}_t2{offline_batch_budget}_o{best_s}_best_j{batch_id}_s{seeds[seed_index]}.npy",
                    allow_pickle=True)
                clusterings[batch_id][seed_index] = clustering
    elif prefix == "pure_online":
        best_s = None
        best_score = np.inf
        best_optimized = None
        for s in range(3):
            param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_pure_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{s}.csv"
            with open(param_path, "r") as param_log_file:
                content = param_log_file.readlines()
                if len(content) > 0:
                    optimized = content[-1]
                    # print(optimized)
                    score = float(optimized.split(";")[2])
                    # print(score)
                    if score < best_score:
                        best_score = score
                        best_s = s
                        best_optimized = optimized
        if best_optimized is None:
            raise Exception("No results")
        print(best_s, best_score, best_optimized.replace("\n", ""), flush=True)


        if online_alg not in non_deterministic_online:
            seeds = [0]
        else:
            seeds = [0,1,2,3,4,5,6,7,8,9]

        for batch in range(batch_ids):
            clusterings[batch] = {}
            for seed_index in range(len(seeds)):
                print(batch, seeds[seed_index])
                seed = seeds[seed_index]

                if seed == 0:
                    clustering = np.load(
                        f"./{label_folder}/{ds}/{online_alg}/clustering_{ds}_pure_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                        allow_pickle=True)
                else:
                    clustering = np.load(
                        f"./replicate_{label_folder}/{ds}/{online_alg}/clustering_{ds}_pure_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                        allow_pickle=True)

                print(clustering)
                clusterings[batch][seed_index] = clustering
    elif prefix == "pure_offline":
        best_s = None
        best_score = np.inf
        best_optimized = None
        for s in range(3):
            param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_pure_{offline_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{s}.csv"
            with open(param_path, "r") as param_log_file:
                content = param_log_file.readlines()
                if len(content) > 0:
                    optimized = content[-1]
                    # print(optimized)
                    score = float(optimized.split(";")[2])
                    # print(score)
                    if score < best_score:
                        best_score = score
                        best_s = s
                        best_optimized = optimized
        if best_optimized is None:
            raise Exception("No results")
        print(best_s, best_score, best_optimized.replace("\n", ""), flush=True)

        if offline_alg == "kmeans":
            seeds = list(range(10))
        else:
            seeds = [0]

        for batch in range(batch_ids):
            clusterings[batch] = {}
            for seed_index in range(len(seeds)):
                print(batch, seeds[seed_index])
                clustering = np.load(
                    f"./{label_folder}/{ds}/{offline_alg}/clustering_{ds}_pure_{offline_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                    allow_pickle=True)

                print(clustering)
                clusterings[batch][seed_index] = clustering
    elif offline == "seed_online":
        best_s = None
        best_score = np.inf
        best_optimized = None
        for s in range(3):
            param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{s}.csv"
            with open(param_path, "r") as param_log_file:
                content = param_log_file.readlines()
                if len(content) > 0:
                    optimized = content[-1]
                    # print(optimized)
                    score = float(optimized.split(";")[2])
                    # print(score)
                    if score < best_score:
                        best_score = score
                        best_s = s
                        best_optimized = optimized
        if best_optimized is None:
            raise Exception("No results")
        print(best_s, best_score, best_optimized.replace("\n", ""), flush=True)

        seeds = [0]
        for batch in range(batch_ids):
            clusterings[batch] = {}
            for seed_index in range(len(seeds)):
                print(batch, seeds[seed_index])
                clustering = np.load(
                    f"./{label_folder}/{ds}/{online_alg}/{label_name}_{ds}_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                    allow_pickle=True)
                print(clustering)
                clusterings[batch][seed_index] = clustering
    else:
        best_s = None
        best_score = np.inf
        best_optimized = None
        for s in range(3):
            param_path = f"{prefix}_param_logs/{ds}/{prefix}_params_{ds}_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{s}.csv"
            with open(param_path, "r") as param_log_file:
                content = param_log_file.readlines()
                if len(content) > 0:
                    optimized = content[-1]
                    # print(optimized)
                    score = float(optimized.split(";")[2])
                    # print(score)
                    if score < best_score:
                        best_score = score
                        best_s = s
                        best_optimized = optimized
        if best_optimized is None:
            raise Exception("No results")
        print(best_s, best_score, best_optimized.replace("\n", ""), flush=True)

        if online_alg not in non_deterministic_online:
            seeds = [0]
        else:
            seeds = [0,1,2,3,4,5,6,7,8,9]

        for batch in range(batch_ids):
            clusterings[batch] = {}
            for seed_index in range(len(seeds)):
                print(batch, seeds[seed_index])
                seed = seeds[seed_index]
                if seed == 0:
                    clustering = np.load(
                        f"./{label_folder}/{ds}/{online_alg}/{label_name}_{ds}_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                        allow_pickle=True)
                else:
                    clustering = np.load(
                        f"./replicate_{label_folder}/{ds}/{online_alg}/{label_name}_{ds}_{online_alg}_b{batch_size}_e{args.eval_size}_t{online_batch_budget}_s{best_s}_best_j{batch}_s{seeds[seed_index]}.npy",
                        allow_pickle=True)

                print(clustering)

                # if offline == "online":
                #     mini_summaries = np.load(
                #     f"./online_mini_summaries/{ds}/{online_alg}/mini_summary_{ds}_{online_alg}_b{batch_size}_t{batch_budget}_s{best_s}_i{best_i}_j{batch}_s{seed}.npy",
                #     allow_pickle=True).item()
                #     if len(mini_summaries["status"]) > 0:
                #         statuses = np.array(mini_summaries["status"])
                #         #print(statuses)
                #         outlie_summaries = np.where(statuses == 'outlie', 1, 0)
                #         sporad_summaries = np.where(statuses == 'sporad', 1, 0)
                #         sporad2_summaries = np.where(statuses == 'sporadic', 1, 0)
                #         deleted_summaries = np.where(statuses == 'deleted', 1, 0)
                #         deleted2_summaries = np.where(statuses == 'delete', 1, 0)
                #
                #         noise_summaries = outlie_summaries + sporad_summaries + sporad2_summaries + deleted_summaries + deleted2_summaries
                #         #print(noise_summaries)
                #
                #         noise_clustering = np.where(noise_summaries[clustering] == 1, -1, clustering)
                #         #print(noise_clustering)
                #         clustering = noise_clustering
                #     else:
                #         clustering = np.ones(len(clustering)) * -1

                clusterings[batch][seed_index] = clustering

    print("clusterings done")
    return clusterings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--online', default="one", type=str, help='Online Summarization Method')
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
    if args.offline == "online":
        store_name = f"result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}.npy"
        store_name_std = f"result_evals/{args.offline}/{args.ds}_{args.online}_{args.batch}_{args.batch_budget_online}_std.npy"

    else:
        store_name = f"result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}.npy"
        store_name_std = f"result_evals/{args.offline}/{args.ds}_{args.online}_{args.offline}_{args.batch}_{args.batch_budget_online}_{args.batch_budget_offline}_std.npy"


    if not os.path.exists("result_evals"):
        os.makedirs("result_evals", exist_ok=True)

    if not os.path.exists(f"result_evals/{args.offline}"):
        os.makedirs(f"result_evals/{args.offline}", exist_ok=True)

    if os.path.exists(store_name) and not args.overwrite:
        raise FileExistsError(store_name)


    data_points, labels = load_data(args.ds)

    batch_ids = math.ceil(len(labels)/args.batch)

    print(batch_ids, flush=True)

    if args.online != "one":
        clusterings = get_clustering(args)
    all_metrics = {}
    all_metrics_std = {}

    full_labels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    min_clu = np.max(labels) +1
    full_min_clu = {0: min_clu, 1: min_clu, 2: min_clu, 3: min_clu, 4: min_clu, 5: min_clu, 6: min_clu, 7: min_clu, 8: min_clu, 9: min_clu}

    for batch_id in range(batch_ids):
        start = args.batch * batch_id
        stop = min(args.batch * (batch_id + 1), len(data_points))
        batch_labels = labels[max(0, start - (args.eval_size*args.batch)):stop]
        batch_data_points = data_points[max(0, start - (args.eval_size*args.batch)):stop]

        # print(get_gt_clusters(batch_labels, batch_data_points))

        batch_size = len(batch_labels)
        if args.online != "one":
            batch_clusterings = clusterings[batch_id]
        else:
            batch_clusterings = {0:np.ones(len(batch_labels))}

        print(np.unique(batch_clusterings, return_counts=True))

        batch_metrics = {
            "ari": [],
            "ami": [],
            "purity": [],
            "offline_score": [],
            "online_score": [],
            "accuracy": [],
            "nmi": [],
            "fowlkes_mallows": [],
            "homogeneity": [],
            "completeness": [],
            "fair_nmi": [],
            "variation_of_information": [],
            "simplified_silhouette": [],
            "disco": [],
            #"cmm": [],
            #"cmm_alt": [],
            "batch_ts": [],
            "clu_num": [],
            "clu_dist": [],
            "noise_ratio": []
        }

        for seed in batch_clusterings.keys():

            batch_cluster_labels = batch_clusterings[seed]


            print(batch_id, seed, len(batch_cluster_labels), np.unique(batch_cluster_labels), flush=True)


            clus, clu_counts = np.unique(batch_cluster_labels, return_counts=True)
            batch_metrics["clu_num"].append(len(clus))
            batch_metrics["clu_dist"].append(clu_counts)

            noise_ids = np.where(clus < 0)
            noise_num = np.sum(clu_counts[noise_ids])
            noise_ratio = noise_num/len(batch_cluster_labels)
            batch_metrics["noise_ratio"].append(noise_ratio)

            ari = adjusted_rand_score(batch_labels, batch_cluster_labels)
            batch_metrics["ari"].append(ari)
            ami = adjusted_mutual_info_score(batch_labels, batch_cluster_labels)
            batch_metrics["ami"].append(ami)
            batch_metrics["offline_score"].append(2 - (ami + ari))
            batch_purity = purity(labels_true=batch_labels, labels_pred=batch_cluster_labels)
            batch_metrics["online_score"].append(1 - batch_purity)
            batch_metrics["purity"].append(batch_purity)

            _, uniq_batch_labels = np.unique(batch_labels, return_inverse=True)
            _, uniq_cluster_labels = np.unique(batch_cluster_labels, return_inverse=True)

            #print(uniq_cluster_labels, uniq_batch_labels, flush=True)

            accuracy = unsupervised_clustering_accuracy(uniq_batch_labels, uniq_cluster_labels)
            batch_metrics["accuracy"].append(accuracy)

            nmi = normalized_mutual_info_score(batch_labels, batch_cluster_labels)
            batch_metrics["nmi"].append(nmi)
            cur_fm = fowlkes_mallows_score(batch_labels, batch_cluster_labels)
            batch_metrics["fowlkes_mallows"].append(cur_fm)
            cur_homo = homogeneity_score(batch_labels, batch_cluster_labels)
            batch_metrics["homogeneity"].append(cur_homo)
            cur_comp = completeness_score(batch_labels, batch_cluster_labels)
            batch_metrics["completeness"].append(cur_comp)
            cur_fnmi = fair_normalized_mutual_information(uniq_batch_labels, uniq_cluster_labels)
            batch_metrics["fair_nmi"].append(cur_fnmi)
            cur_voi = variation_of_information(uniq_batch_labels, uniq_cluster_labels)
            batch_metrics["variation_of_information"].append(cur_voi)
            #cur_itevcm = information_theoretic_external_cluster_validity_measure(uniq_batch_labels, uniq_cluster_labels)
            #batch_metrics["information_theoretic_external_cluster_validity_measure"].append(cur_itevcm)

            # if len(np.unique(batch_cluster_labels)) == len(batch_cluster_labels):
            #     cur_db = 0
            # elif len(np.unique(batch_cluster_labels)) > 1:
            #     cur_db = davies_bouldin_score(batch_data_points, batch_cluster_labels)
            # else:
            #     max_db = 0  # undefined -> worst case, but uncapped -> use max 1 vs all score
            #     for i in range(len(batch_labels)):
            #         fake_labels = [0] * len(batch_labels)
            #         fake_labels[i] = 1
            #         dbs = davies_bouldin_score(batch_data_points, fake_labels)
            #         if max_db < dbs:
            #             max_db = dbs
            #     cur_db = max_db
            # batch_metrics["davies_bouldin_score"].append(cur_db)
            #
            # if len(np.unique(batch_cluster_labels)) == len(batch_cluster_labels):
            #     cur_ch = 0
            # elif len(np.unique(batch_cluster_labels)) > 1:
            #     cur_ch = calinski_harabasz_score(batch_data_points, batch_cluster_labels)
            # else:
            #     max_ch = 0  # undefined -> worst case, but uncapped -> use max 1 vs all score
            #     for i in range(len(batch_data_points)):
            #         fake_labels = [0] * len(batch_data_points)
            #         fake_labels[i] = 1
            #         ch = calinski_harabasz_score(batch_data_points, fake_labels)
            #         if max_ch < ch:
            #             max_ch = ch
            #     cur_ch = max_ch
            # batch_metrics["calinski_harabasz_score"].append(cur_ch)

            # if len(np.unique(batch_cluster_labels)) == len(batch_cluster_labels) or len(np.unique(batch_cluster_labels)) == 1:
            #     max_sdbw = 0  # undefined -> worst case, but uncapped -> use max 1 vs all score
            #     for i in range(len(batch_data_points)):
            #         fake_labels = [0] * len(batch_data_points)
            #         fake_labels[i] = 1
            #         sdbw = S_Dbw(batch_data_points, fake_labels, centers_id=None, method='Tong', alg_noise='comb',
            #                  centr='mean', nearest_centr=True, metric='euclidean')
            #         if max_sdbw < sdbw:
            #             max_sdbw = sdbw
            #     cur_sdbw = max_sdbw
            # else:
            #     cur_sdbw = S_Dbw(batch_data_points, batch_cluster_labels, centers_id=None, method='Tong', alg_noise='comb',
            #                  centr='mean', nearest_centr=True, metric='euclidean')
            # batch_metrics["s_dbw"] = cur_sdbw


            cur_ssil = simplified_silhouette_score(batch_data_points, batch_cluster_labels)
            batch_metrics["simplified_silhouette"].append(cur_ssil)

            # DISCO disabled
            batch_disco = disco_score(batch_data_points, batch_cluster_labels, min_points=5)
            batch_metrics["disco"].append(batch_disco)



            #cmm, cmm_alt = get_cmm(x=batch_data_points, gt=batch_labels, clu=batch_cluster_labels, k=5, w=[1]*len(batch_labels), alternate=True)
            #batch_metrics["cmm"].append(cmm)
            #batch_metrics["cmm_alt"].append(cmm_alt)
            timesteps = np.arange(len(batch_labels))
            _, _, TS_100_1000 = tempsil(timesteps, batch_data_points, np.array(batch_cluster_labels), s=100, kn=1000, c=1)
            if math.isnan(TS_100_1000): # error -> replace with worst-case
                TS_100_1000 = -1
            batch_metrics["batch_ts"].append(TS_100_1000)
            mapping = {}
            mapping_overlap = {}
            rev_mapping = {}
            rev_mapping_overlap = {}
            for label in np.unique(batch_labels):
                rev_mapping_overlap[label] = 0
            for clu in np.unique(batch_cluster_labels):
                clu_ids = np.where(batch_cluster_labels == clu)[0]
                max_overlap = 0
                for label in np.unique(batch_labels):
                    label_ids = np.where(batch_labels == label)[0]
                    overlap = len(np.intersect1d(clu_ids, label_ids))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        mapping[clu] = label
                        mapping_overlap[clu] = overlap
                label_max_overlap = rev_mapping_overlap[mapping[clu]]
                if label_max_overlap < mapping_overlap[clu]:
                    rev_mapping[mapping[clu]] = clu
                    rev_mapping_overlap[mapping[clu]] = mapping_overlap[clu]
            assignment = {}
            for clu in np.unique(batch_cluster_labels):
                if clu == -1:
                    assignment[clu] = -1
                else:
                    if clu == rev_mapping[mapping[clu]]:
                        assignment[clu] = mapping[clu]
                    else:
                        full_min_clu[seed] += 1
                        assignment[clu] = full_min_clu[seed]
            for pred in batch_cluster_labels:
                full_labels[seed].append(assignment[pred])

        avg_batch_metrics = {}
        std_batch_metrics = {}
        for metric_key in batch_metrics.keys():
            if metric_key != "clu_dist":
                avg_batch_metrics[metric_key] = float(np.mean(batch_metrics[metric_key]))
                if len(batch_metrics[metric_key]) > 1:
                    std_batch_metrics[metric_key] = float(np.std(batch_metrics[metric_key]))
                else:
                    std_batch_metrics[metric_key] = np.nan
        avg_batch_metrics["weight"] = len(batch_labels)
        std_batch_metrics["weight"] = 0
        all_metrics[batch_id] = avg_batch_metrics
        all_metrics_std[batch_id] = std_batch_metrics


    all_metrics["all"] = {}
    all_metrics["all_std"] = {}

    print(all_metrics)
    print(all_metrics_std)
    for metric in all_metrics[0].keys():
        metric_sum = 0
        metric_std_sum = 0
        weight_sum = 0
        for batch_id in range(batch_ids):
            print(batch_id)
            metric_sum += all_metrics[batch_id][metric] * all_metrics[batch_id]["weight"]
            metric_std_sum += all_metrics_std[batch_id][metric] * all_metrics[batch_id]["weight"]
            weight_sum += all_metrics[batch_id]["weight"]
        all_metrics["all"][metric] = metric_sum / weight_sum
        all_metrics["all_std"][metric] = metric_std_sum / weight_sum

    # ts_list = []
    # for full_label in full_labels.values():
    #     print("full_label", len(full_label))
    #     if len(full_label) >= 1:
    #         _, _, TS = tempsil(np.arange(len(full_label)), data_points, np.array(full_label), s=100, kn=1000, c=1)
    #         ts_list.append(TS)
    # all_metrics["all"]["labelmatch_ts"] = float(np.mean(ts_list))

    pprint(all_metrics)

    np.save(store_name, all_metrics)
    np.save(store_name_std, all_metrics_std)

