import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpt
from scipy import stats

from defines import main_online_algorithms, online_colors, online_method_names, real_datasets


def make_rank_change_figure(first, first_score, second, second_score):

    first_ranks = {}
    with (open(f"result_extracts/ranks_{first}_main_{first_score}.txt", "r") as f):
        lines = f.readlines()
        for line in lines:
            if "dataset_name" not in line:
                line.replace("\n", "")
                line_split = line.split(",")
                key = line_split[0]
                value = float(line_split[2])
                if key not in first_ranks:
                    first_ranks[key] = []
                first_ranks[key].append(value)

    mean_first_rank = {}
    for key in first_ranks:
        mean_first_rank[key] = np.mean(first_ranks[key])


    second_ranks = {}
    with (open(f"result_extracts/ranks_{second}_main_{second_score}.txt", "r") as f):
        lines = f.readlines()
        for line in lines:
            if "dataset_name" not in line:
                line.replace("\n", "")
                line_split = line.split(",")
                key = line_split[0]
                value = float(line_split[2])
                if key not in second_ranks:
                    second_ranks[key] = []
                second_ranks[key].append(value)

    mean_second_rank = {}
    for key in second_ranks:
        mean_second_rank[key] = np.mean(second_ranks[key])

    print(mean_first_rank)
    print(mean_second_rank)

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.figure(figsize=(12, 2))
    for i in range(len(main_online_algorithms)):
        online = main_online_algorithms[i]
        method_ranks = []

        key = online_method_names[online]

        black = 'black'

        plt.boxplot(second_ranks[key], tick_labels=[key], positions=[i], widths=[0.5], patch_artist=True,
                    boxprops=dict(facecolor=online_colors[online], color=black),
                    capprops=dict(color=black),
                    whiskerprops=dict(color=black),
                    flierprops=dict(color=black, markeredgecolor=black),
                    medianprops=dict(color=black))

        if mean_first_rank[key] < mean_second_rank[key]:
            color = 'red'
        else:
            color = 'blue'

        diff = mean_second_rank[key] - mean_first_rank[key]
        abs_diff = abs(diff)
        arrow = mpt.FancyArrow(i-0.4, mean_first_rank[key], 0, diff, color=color, length_includes_head=True,
                               head_length = min(0.3, abs_diff/3), head_width= min(0.1, max(abs_diff/8, 0.04)), width=0.02)
        plt.gca().add_patch(arrow)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Relative Rank")

    plt.ylim(0, 22)
    plt.savefig(f"result_figures/result_boxplot_{first}_{first_score}_{second}_{second_score}.pdf", bbox_inches='tight', transparent=True)
    plt.close()

    # first_ranks_list = []
    # second_ranks_list = []
    #
    #
    # for i in range(len(main_online_algorithms)):
    #     online = main_online_algorithms[i]
    #     key = online_method_names[online]
    #     rng = np.random.default_rng()
    #     method = stats.PermutationMethod(random_state=rng)
    #     res = stats.pearsonr(first_ranks[key], second_ranks[key], method=method)
    #
    #     print(key, res)
    #     first_ranks_list.append(first_ranks[key])
    #     second_ranks_list.append(second_ranks[key])
    #
    # first_ranks_np = np.array(first_ranks_list)
    # second_ranks_np = np.array(second_ranks_list)
    #
    # first_ranks_np, second_ranks_np = first_ranks_np.T, second_ranks_np.T
    # print("---")
    # for i in range(len(real_datasets)):
    #     rng = np.random.default_rng()
    #     method = stats.PermutationMethod(random_state=rng)
    #     res = stats.pearsonr(first_ranks_np[i], second_ranks_np[i], method=method)
    #     print(real_datasets[i], res)








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', default="seed_online", type=str, help='First ranked aspect')
    parser.add_argument('--first_score', default="real_completed_purity", type=str, help='Score for ranking of first aspect')
    parser.add_argument('--second', default="shade", type=str, help='Second ranked aspect')
    parser.add_argument('--second_score', default="real_completed_offline_score", type=str, help='Score for ranking of second aspect')
    args = parser.parse_args()


    make_rank_change_figure(args.first, args.first_score, args.second, args.second_score)
