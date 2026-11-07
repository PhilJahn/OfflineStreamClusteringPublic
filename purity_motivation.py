import os

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import sklearn.neighbors as skn


from datahandler import load_data
from subset_methods import BatchSummarizer


def summarize_data(data, weight, summary_centers):
    nn = skn.NearestNeighbors(n_neighbors=1).fit(summary_centers)
    nn_dists, neighbors = nn.kneighbors(data)

    #print(neighbors)

    actual_positions = []
    actual_weights = []
    actual_stds = []
    actual_radii = []

    weight_2 = [[weight[i],weight[i]] for i in range(len(weight))]
    weight_2 = np.array(weight_2)

    for i in range(len(summary_centers)):
        included = np.where(neighbors == i)[0]
        print(i, included)
        actual_weight = np.sum(weight[included])
        actual_position = np.sum(weight_2[included] * data[included],axis=0) / actual_weight
        actual_std = np.sqrt((np.sum((weight_2[included] * data[included])**2,axis=0) / actual_weight) - (actual_position ** 2))
        true_std = np.std(data[included], axis=0)
        actual_positions.append(actual_position)
        actual_weights.append(actual_weight)
        actual_stds.append(actual_std)
        actual_radius = np.sum(actual_std**2) ** 0.5
        actual_radii.append(actual_radius)

    return actual_positions, actual_weights, actual_stds, actual_radii


def build_figure(name, stale_data, actual_batch_data, actual_batch_labels, centers, weights, radii):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = np.array(prop_cycle.by_key()['color'])
    plt.figure(figsize=(3,3))
    plt.scatter(actual_batch_data[:, 0], actual_batch_data[:, 1], s=4, c=actual_batch_labels, alpha=0.25)
    plt.scatter(stale_data[:, 0], stale_data[:, 1], s=4, c="lightgray", alpha=0.1)

    true_nn = skn.NearestNeighbors(n_neighbors=1).fit(centers)
    _, actual_neighbors = true_nn.kneighbors(actual_batch_data)
    purity_sum = 0
    for l in range(len(centers)):
        # print(mc.mean)
        mc_patch = ptc.Circle((float(centers[l][0]), float(centers[l][1])), radii[l], alpha=0.2, color="black")
        plt.gca().add_patch(mc_patch)
        plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l], c="black", marker="X", linewidths=1,
                    edgecolors="black")
        included = np.where(actual_neighbors == l)[0]
        print(included)
        unique_labels, counts = np.unique(actual_batch_labels[included], return_counts=True)
        if len(counts) >= 1:
            summary_purity = np.max(counts) / np.sum(counts)
            purity_sum += summary_purity * np.sum(counts)


    purity = purity_sum / len(actual_batch_data)
    plt.title(f"Purity: {purity:.1f}")

    plt.axis('equal')
    plt.savefig(f"purity_figure/{name}_purity.pdf", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    data, labels = load_data("stream_densired")
    os.makedirs("purity_figure", exist_ok=True)
    print(data, labels)

    actual_batch_data = data[4000:5000]
    actual_batch_labels = labels[4000:5000]
    input_datas = [data[4000:5000], data[3000:5000], data[4500:5000]]

    stale_data = input_datas[1][:-1000]

    input_data_names = ["full", "stale", "half"]
    summary_counts = [20, 1000]
    for summary_count in summary_counts:
        for i in range(len(input_datas)):
            input_data = input_datas[i]

            batch_sampling = BatchSummarizer("batch_kcentroid", summary_count)
            batch_sampling.learn(input_data, list(range(1000)))
            batch_summary = batch_sampling.get_summary()
            summary_centers = batch_summary["centers"]

            weighting = [0.5 ** (i // 1000) for i in range(len(input_data))]
            weighting = np.flip(weighting)
            print(weighting)

            centers, weights, stds, radii = summarize_data(input_data, weighting, summary_centers)
            build_figure(f"{summary_count}_{input_data_names[i]}", stale_data, actual_batch_data, actual_batch_labels, centers, weights, radii)