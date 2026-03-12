import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as ptc

import utils
from datahandler import load_data


online_method_names = {"full_batch": "Full Batch", "random": "Random", "lwc": "LWC-M", "lwc_square": "LWC", "kcentroid": "k-Centroid", "protras": "ProTraS",
                       "streamkmeans+": "STREAMKMeans (ours)", "streamkmeans_river": "StreamKMeans", "streamkmplus+": "StreamKM++", "birch_sklearn": "BIRCH", "clustream": "CluStream (ours)", "clustream_river": "CluStream",
                       "denstream": "DenStream (ours)", "denstream_river": "DenStream", "dstream": "D-Stream", "swclustering+": "SWClustering", "gbfuzzystream_original": "GB-FuzzyStream", "gbfuzzystream": "GB-FuzzyStream (ours)",
                       "stmstream+": "STM-Stream", "emcstream": "EMCStream (ours)", "emcstream_original": "EMCStream", "pure_offline": "Full Data"
                       }

def fig_maker(ds, online_alg, offline_alg, batches):
    np.random.seed(0)
    stream, labels = load_data(ds)
    best_score = 2

    time = 1800
    batch_size = 1000
    plt.rcParams.update({'font.size': 37})

    for s in range(3):
        param_path = f"online_param_logs/{ds}/online_params_{ds}_{online_alg}_b1000_t{time}_s{s}.csv"
        with open(param_path, "r") as param_log_file:
            optimized = param_log_file.readlines()[-1]
            #print(optimized)
            score = float(optimized.split(";")[2])
            #print(score)
            if score < best_score:
                best_score = score
                best_i = int(optimized.split("i")[-1])
                best_s = s
                best_optimized = optimized

    batch_ids = np.arange(0, len(stream), step=batch_size).tolist()
    for h in batches:
        print(h, len(stream[batch_ids[h]:batch_ids[h + 1]]))
        stream_batch = stream[batch_ids[h]:batch_ids[h + 1]]
        label_batch = labels[batch_ids[h]:batch_ids[h + 1]]

        # print(timestamp_batch)
        # https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
        min_store = np.load(
            f"./online_mini_summaries/{ds}/{online_alg}/mini_summary_{ds}_{online_alg}_b1000_t{time}_s{best_s}_i{best_i}_j{h}_s0.npy",
            allow_pickle=True)
        min_store = min_store.item()

        if offline_alg == "none":
            clustering = np.load(
                f"./online_assignments/{ds}/{online_alg}/assignment_{ds}_{online_alg}_b1000_t{time}_s{best_s}_i{best_i}_j{h}_s0.npy",
                allow_pickle=True)
            #print(len(clustering))
            summary_clustering = np.arange(len(min_store["center"]))
            status = min_store["status"]
            for i in range(len(summary_clustering)):
                if status[i] in ["outlie", "sporad", "sporadic", "deleted", "delete"]:
                    summary_clustering[i] = -1


        elif offline_alg == "adjacency":
            clustering = np.load(
                f"./online_pseudoclusterings/{ds}/{online_alg}/pseudoclustering_{ds}_{online_alg}_b1000_t{time}_s{best_s}_i{best_i}_j{h}_s0.npy",
                allow_pickle=True)
            positions = min_store["center"]
            radii = min_store["radii"]
            status = min_store["status"]
            mc_dists = utils.distance_calc(positions, positions, dist="euclidean")
            two_radii = np.array([radii + radii[i] for i in range(len(positions))])
            connected = np.where(mc_dists <= two_radii, True, False)
            partitions = np.arange(len(positions) + 1)  # get connected
            for i in range(len(partitions) - 1):
                if (status[i] != "outlie" and status[i] != "sporad" and status[i] != "sporadic" and
                        status[i] != "deleted" and status[i] != "delete"):
                    for j in range(i):
                        if (status[j] != "outlie" and status[j] != "sporad" and status[j] != "sporadic" and
                                status[j] != "deleted" and status[j] != "delete"):
                            if connected[i, j]:
                                if partitions[i] != partitions[j]:
                                    if status[i] == "dense" and status[j] == "dense":
                                        key = min(partitions[i], partitions[j])
                                        other_key = max(partitions[i], partitions[j])
                                        partitions[np.where(partitions == other_key)[0]] = key
                                    elif status[i] == "normal" and status[j] == "dense":
                                        partitions[i] = partitions[j]
                                    elif status[i] == "dense" and status[j] == "normal":
                                        partitions[j] = partitions[i]
                else:
                    partitions[i] = -1
            partitions[-1] = -1
            summary_clustering = partitions
        else:
            raise NotImplementedError
        #clu_num = len(np.unique(summary_clustering))
        clu_num = np.max(summary_clustering) + 2

        if clu_num <= 10:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = np.array(prop_cycle.by_key()['color'])
            np.random.shuffle(colors)
            colors = colors.tolist()
            colors.append("#808080")
            colors = np.array(colors)
        elif clu_num <= 20:
            # https://stackoverflow.com/questions/57845444/how-to-convert-a-listedcolormap-to-a-list-of-colors
            colors = np.array(plt.get_cmap('tab20').colors)
            np.random.shuffle(colors)
            colors = colors.tolist()
            colors.append(np.array([0.5, 0.5, 0.5, 1]))
            colors = np.array(colors)
        else:
            cmap = plt.get_cmap('jet')
            #print(cmap(0.5))
            #https://stackoverflow.com/questions/66171467/how-to-get-all-color-codes-from-linearsegmentedcolormap
            colors = np.array(list(set([cmap(a) for a in np.linspace(0.05, 0.95, clu_num)])))
            np.random.shuffle(colors)
            colors = colors.tolist()
            colors.append(np.array([0.5, 0.5, 0.5, 1]))
            colors = np.array(colors)
            #print(colors)
            #.by_key()['color']

        #print(len(colors))

        #print(np.unique(summary_clustering))
        #print(np.unique(clustering))

        plt.figure(figsize=(10, 10))

        if offline_alg != "dstream":
            centers = min_store["center"]
            weights = min_store["weights"]
            radii = min_store["radii"]
            status = min_store["status"]
        else:
            centers = min_store["center"][:-1]
            weights = min_store["weights"][:-1]
            radii = min_store["radii"][:-1]
            status = min_store["status"][:-1]

        plt.scatter(x=stream_batch[:, 0], y=stream_batch[:, 1], c=colors[clustering % len(colors)], s=10)

        for l in range(len(min_store["center"])):

            if summary_clustering[l] == -1:
                color = "lightgrey"
            else:
                color = colors[summary_clustering[l] % len(colors)]

            #print(weights[l], radii[l])

            if online_alg != "dstream":
                mc_patch = ptc.Circle((float(centers[l][0]), float(centers[l][1])), radii[l], alpha=0.4, color=color)
            else:
                mc_patch = ptc.Rectangle((centers[l] - radii[l]), 2 * radii[l], 2 * radii[l], alpha=0.4, color=color)
            plt.gca().add_patch(mc_patch)

            if status[l] == "dense":
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l] * 10,
                        color=color,
                        marker="X", linewidths=1, edgecolors="black")
            elif status[l] == "normal":
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l] * 10,
                        color=color,
                        marker="P", linewidths=1, edgecolors="black")
            else:
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l] * 10,
                        color=color,
                        marker=".", linewidths=1, edgecolors="black")
        #plt.gca().axes.get_yaxis().set_visible(False)
        #plt.gca().axes.get_xaxis().set_visible(False)
        ax = plt.gca()
        #ax.axis('off')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        plt.title(online_method_names[online_alg])
        plt.savefig(f"figure1_content/{ds}_{online_alg}_{offline_alg}_{h}.jpg", bbox_inches='tight')
        plt.close()
        #raise Exception


if __name__ == "__main__":
    dataset = "stream_densired"
    #online_alg = "dstream"
    offline_alg = "adjacency"
    for online_alg in ["clustream", "clustream_river", "random", "protras", "dstream", "denstream", "denstream_river", "gbfuzzystream", "gbfuzzystream_original",
                       "lwc_square", "kcentroid", "stmstream+", "swclustering+", "streamkmplus+", "streamkmeans+", "streamkmeans_river", "birch_sklearn"]:
        fig_maker(dataset, online_alg, offline_alg, [5])
