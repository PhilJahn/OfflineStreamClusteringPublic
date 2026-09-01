import argparse

import numpy as np
from matplotlib import pyplot as plt

from clustpy.metrics import purity
from datahandler import load_data
from optimize_offline import get_summary_id, get_batch_summary

def get_offline_summary_id(args):
    prefix = "offline"
    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_t1{args.online_batch_budget}_t2{args.offline_batch_budget}"


    best_i = None
    best_score = np.inf
    offline_param_path = f"{prefix}_param_logs/{args.ds}/{args.online}/{args.offline}/{off_prefix}_params_{name}_c{args.batch_id}_o{args.smac_seed}.csv"
    with open(offline_param_path, "r") as offline_param_log_file:
        optimized = offline_param_log_file.readlines()[-1]
        #print(optimized)
        score = float(optimized.split(";")[2])
        #print(score)
        if score < best_score:
            best_i = int(optimized.split("i")[-1])
    return best_i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="stream_densired", type=str, help='Dataset')
    parser.add_argument('--online', default="gbfuzzystream_original", type=str, help='Online Summarization Method')
    parser.add_argument('--offline', default="adjacency", type=str, help='Offline Clustering Method')
    parser.add_argument('--online_batch_budget', default="1_logdim"
                        , type=str, help='Online SMAC AutoML Budget (in seconds per batch)')
    parser.add_argument('--offline_batch_budget', default="1_logdim"
                        , type=str, help='Offline SMAC AutoML Budget (in seconds per batch)')

    parser.add_argument('--eval_size', default=1, type=int, help='Number of preceding batches to include in evaluation')
    parser.add_argument('--batch_id', default=3, type=int, help='Batch to show')
    parser.add_argument('--batch', default=1000, type=int, help='Batch Size for Stream Evaluation')
    parser.add_argument('--smac_seed', default=42, type=int, help='Seed for SMAC Optimization')
    parser.add_argument('--offline_seed', default=0, type=int, help='Seed for Offline Algorithm')

    args = parser.parse_args()


    data_points, labels = load_data(args.ds, 0)
    batch_data = data_points[max(0, args.batch_id*args.batch - (args.eval_size * args.batch)):(args.batch_id + 1)
                                                                                         *args.batch]
    batch_labels = labels[max(0, args.batch_id*args.batch - (args.eval_size * args.batch)):(args.batch_id + 1)
                                                                                         *args.batch]
    print(batch_data.shape)

    batch_num = int(np.ceil(len(data_points) / args.batch))
    best_online_smac_seed, best_incumbent_id, best_nondet_seed = get_summary_id(args)


    batch_summary_centers, batch_summary_radii, batch_summary_weights, batch_summary_status, batch_assignments = (
        get_batch_summary(args, args.batch_id, best_online_smac_seed, best_incumbent_id, best_nondet_seed))

    off_prefix = "offline"
    label_prefix = f"{off_prefix}_labels/{args.ds}/{args.online}/{args.offline}"
    summary_label_prefix = f"{off_prefix}_summary_labels/{args.ds}/{args.online}/{args.offline}"
    offline_id = get_offline_summary_id(args)
    name = f"{args.ds}_{args.online}_{args.offline}_b{args.batch}_t1{args.online_batch_budget}_t2{args.offline_batch_budget}"
    clustering = np.load(f"{label_prefix}/clustering_{name}_i{offline_id}_j{args.batch_id}_s{args.offline_seed}.npy")
    summary_clustering = np.load(f"{summary_label_prefix}/summary_clustering_{name}_i{offline_id}_j{args.batch_id}_s{args.offline_seed}.npy")
    print(clustering)

    print(purity(batch_labels, clustering))
    plt.figure(figsize=(10,10))
    plt.scatter(batch_data[:, 0], batch_data[:, 1], c=clustering)
    plt.show()
    plt.figure(figsize=(10,10))

    indices_dense = np.where(batch_summary_status == "dense")[0]
    indices_normal = np.where(batch_summary_status == "normal")[0]


    print(max(summary_clustering))
    cmap = plt.get_cmap('jet')
    # print(cmap(0.5))
    # https://stackoverflow.com/questions/66171467/how-to-get-all-color-codes-from-linearsegmentedcolormap
    colors = np.array(list(set([cmap(a) for a in np.linspace(0.05, 0.95, max(summary_clustering))])))
    np.random.shuffle(colors)
    colors = colors.tolist()
    while len(colors) < max(summary_clustering):
        colors.extend(colors)
    colors.append(np.array([0.5, 0.5, 0.5, 1]))
    colors = np.array(colors)

    print(len(colors))

    plt.scatter(batch_summary_centers[indices_dense, 0], batch_summary_centers[indices_dense, 1], c=colors[summary_clustering[indices_dense]], marker="+")
    plt.scatter(batch_summary_centers[indices_normal, 0], batch_summary_centers[indices_normal, 1], c=colors[summary_clustering[indices_normal]], marker="o")

    plt.show()

