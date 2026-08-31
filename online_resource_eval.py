import argparse
import copy
import os
import tracemalloc

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import utils
from online_config_handler import get_online_configspace, batch_sampling_methods, window_sampling_methods
from online_handler import get_online_method
from summary_handler import process_summary

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--online', default="birch_sklearn_2", type=str, help='Online Summarization Method')
    parser.add_argument('--n', default=20000, type=int, help='Instance Count')
    parser.add_argument('--d', default=2, type=int, help='Dimensionality')
    parser.add_argument('--k', default=100, type=int, help='Cluster Count')
    parser.add_argument('--data_seed', default=0, type=int, help='Seed for data generation')
    parser.add_argument('--method_seed', default=0, type=int, help='Seed for method')
    parser.add_argument('--overwrite', default=0, type=int, help='Overwrite existing result')
    args = parser.parse_args()
    args.ds = "blobs"
    args.offline = None
    args.batch = 1000
    args.eval_size = 1
    args.batch_budget = None
    args.overwrite = args.overwrite == 1

    store_name = f"resource_eval/{args.online}_n{args.n}_d{args.d}_k{args.k}_f{args.data_seed}_s{args.method_seed}.txt"

    if os.path.exists(store_name) and not args.overwrite:
        raise FileExistsError(store_name)

    data_points, labels = make_blobs(n_samples=args.n, n_features=args.d, centers=args.k, random_state=args.data_seed)
    timestamps = np.arange(args.n)
    scaler = MinMaxScaler()
    data_points = scaler.fit_transform(data_points)

    x_matrix = np.array([[0,1], [1,0]])
    x_vector = np.array([[0,1]])
    dist1 = utils.distance_calc(x_vector, x_vector, "euclidean")
    dist2 = utils.distance_calc(x_matrix, x_vector, "euclidean")
    dist3 = utils.distance_calc(x_matrix, x_matrix, "euclidean")

    online_method = args.online

    if online_method == "birch_sklearn_2":
        configspace = get_online_configspace("birch_sklearn", data_points, labels, args)
    else:
        configspace = get_online_configspace(online_method, data_points, labels, args)
    default_config_dict = dict(configspace.get_default_configuration())

    if online_method == "birch_sklearn" or online_method == "birch_sklearn_2":
        default_config_dict["max_leaf_entry_num"] = args.k
    elif online_method == "streamkmplus":
        default_config_dict["m"] = args.k
    elif online_method == "stmstream":
        default_config_dict["k"] = args.k
    elif online_method == "swclustering" or online_method == "clustream_river":
        default_config_dict["mmc"] = args.k
    elif online_method in batch_sampling_methods or online_method in window_sampling_methods:
        default_config_dict["sampling_size"] = args.k
    elif online_method == "streamkmeans_river":
        default_config_dict["prototype_num"] = args.k



    print(default_config_dict)
    if online_method == "birch_sklearn_2":
        online_clusterer = get_online_method("birch_sklearn", default_config_dict, args.method_seed)
    else:
        online_clusterer = get_online_method(online_method, default_config_dict, args.method_seed)
    online_clusterer.track_memory = True

    batch_ids = np.arange(0, len(data_points), step=args.batch).tolist()
    batch_ids.append(len(data_points))

    batch_summaries = []
    batch_min_stores = []
    batch_nums = []


    if online_method == "emcstream_original":  # needs to run through fully
        _, _, full_mems_output, full_times_output = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
        full_summary = online_clusterer.get_summary()
        full_mems = [full_mems_output]
        full_times = [full_times_output]
        for batch in range(len(batch_ids) - 1):
            batch_summary = {"embedding": full_summary["embedding"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "timestamps": full_summary["timestamps"][batch_ids[batch]:batch_ids[batch + 1]],
                                   "labels": full_summary["labels"][batch_ids[batch]:batch_ids[batch + 1]]}
            batch_summaries.append(batch_summary)
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            _, min_store = process_summary(online_method, batch_summary, stream_batch, data_points[:10],
                                                    batch_ids[batch], do_pseudoclustering=False)
            batch_min_stores.append(min_store)
            batch_nums.append(len(min_store["center"]))


    elif online_method == "gbfuzzystream_original":  # needs to run through fully, but does have batches internally
        _, _, full_mems_output, full_times_output = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
        full_summary, _ = online_clusterer.get_summary()
        full_mems = []
        full_times = []
        for batch in range(len(batch_ids) - 1):
            full_mems.append([full_mems_output[batch_ids[batch + 1]]])
            full_times.append([full_times_output[batch_ids[batch + 1]]])
            batch_summary = full_summary[batch_ids[batch + 1]]
            batch_summaries.append(batch_summary)
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            _, min_store = process_summary(online_method, batch_summary, stream_batch, data_points[:10],
                                                    batch_ids[batch], do_pseudoclustering=False)
            batch_min_stores.append(min_store)
            batch_nums.append(len(min_store["center"]))
    elif online_method == "birch_sklearn":

        for t in np.linspace(0, args.d/2, 500*args.d):
            print(t)
            default_config_dict["threshold"] = t
            online_clusterer = get_online_method(online_method, default_config_dict, args.method_seed)
            online_clusterer.track_memory = True
            try:
                _, _, full_mems_output, full_times_output = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
            except:
                continue
            break
        full_summary, _ = online_clusterer.get_summary()
        full_mems = []
        full_times = []
        for batch in range(len(batch_ids) - 1):
            full_mems.append([full_mems_output[batch_ids[batch + 1]]])
            full_times.append([full_times_output[batch_ids[batch + 1]]])
            batch_summary = full_summary[batch_ids[batch + 1]]
            batch_summaries.append(batch_summary)
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            _, min_store = process_summary(online_method, batch_summary, stream_batch, data_points[:10],
                                                    batch_ids[batch], do_pseudoclustering=False)
            batch_min_stores.append(min_store)
            batch_nums.append(len(min_store["center"]))
    elif online_method == "birch_sklearn_2":
        t = 0
        while t < args.d:
            print(t)
            default_config_dict["threshold"] = t
            online_clusterer = get_online_method("birch_sklearn", default_config_dict, args.method_seed)
            online_clusterer.track_memory = True
            try:
                _, _, full_mems_output, full_times_output = online_clusterer.learn(data_points.tolist(), timestamps.tolist())
            except:
                t = t + 0.01
                continue
            break
        full_summary, _ = online_clusterer.get_summary()
        full_mems = []
        full_times = []
        for batch in range(len(batch_ids) - 1):
            full_mems.append([full_mems_output[batch_ids[batch + 1]]])
            full_times.append([full_times_output[batch_ids[batch + 1]]])
            batch_summary = full_summary[batch_ids[batch + 1]]
            batch_summaries.append(batch_summary)
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            if online_method == "birch_sklearn_2":
                _, min_store = process_summary("birch_sklearn", batch_summary, stream_batch, data_points[:10],
                                                        batch_ids[batch], do_pseudoclustering=False)
            else:
                _, min_store = process_summary(online_method, batch_summary, stream_batch, data_points[:10],
                                                        batch_ids[batch], do_pseudoclustering=False)
            batch_min_stores.append(min_store)
            batch_nums.append(len(min_store["center"]))
    else:
        full_mems = []
        full_times = []
        for batch in range(len(batch_ids) - 1):
            stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
            timestamp_batch = timestamps[batch_ids[batch]:batch_ids[batch + 1]]
            _, _, batch_mems, batch_times = online_clusterer.learn(stream_batch.tolist(),
                                                                           timestamp_batch.tolist())

            full_mems.append(batch_mems)
            full_times.append(batch_times)
            batch_summary = online_clusterer.get_summary()
            batch_summaries.append(batch_summary)
            _, min_store = process_summary(online_method, batch_summary, stream_batch, data_points[:10],
                                                    batch_ids[batch], do_pseudoclustering=False)
            batch_min_stores.append(min_store)
            batch_nums.append(len(min_store["center"]))

    full_time = np.sum(full_times)
    full_summary_mems = []
    full_min_store_mems = []

    for batch_summary in batch_summaries:
        tracemalloc.start()
        copy.deepcopy(batch_summary)
        mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        full_summary_mems.append(mem)

    for batch_min_store in batch_min_stores:
        tracemalloc.start()
        copy.deepcopy(batch_min_store)
        mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        full_min_store_mems.append(mem)
    if online_method == "emcstream_original":
        full_mem = full_mems[0]
    else:
        full_mem = np.max(full_mems[1:])
    full_summary_mem = np.max(full_summary_mems)
    full_min_store_mem = np.max(full_min_store_mems)
    full_num = np.max(batch_nums)

    print(full_mems)

    print(full_times)

    print(full_mem, full_summary_mem, full_min_store_mem, full_time, full_num)
    result = "full_mem,full_summary_mem,full_min_store_mem,full_time,full_num\n"
    result += f"{full_mem},{full_summary_mem},{full_min_store_mem},{full_time},{full_num}"
    if not os.path.exists("resource_eval"):
        os.makedirs("resource_eval", exist_ok=True)


    with open(store_name, "w") as file:
        file.write(result)

    print("done")