import numpy as np
from matplotlib import pyplot as plt

from defines import main_online_algorithms, online_method_names, online_colors, offline_colors


def make_bar_plot():

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    scores = {}

    best_offline = {}
    for offline_method in ["adjacency", "dbscan", "dpc", "kmeans", "shade"]:
        with (open(f"result_extracts/scores_{offline_method}_main_real_completed_offline_score.txt", "r") as f):
            lines = f.readlines()
            for line in lines:
                if "dataset_name" not in line:
                    line.replace("\n", "")
                    line_split = line.split(",")
                    online = line_split[0]
                    dataset = line_split[1]
                    value = float(line_split[2])

                    if online not in scores:
                        scores[online] = {}
                        best_offline[online] = {}
                    if dataset not in scores[online]:
                        scores[online][dataset] = {}
                        best_offline[online][dataset] = "none"

                    scores[online][dataset][offline_method] = value
                    if best_offline[online][dataset] != "none":
                        if scores[online][dataset][best_offline[online][dataset]] < value:
                            best_offline[online][dataset] = offline_method
                    else:
                        best_offline[online][dataset] = offline_method

    plt.figure(figsize=(10, 2))

    online_names = []
    for i in range(len(main_online_algorithms)):
        online = main_online_algorithms[i]
        online_name = online_method_names[online]

        ctr = {"kmeans": 0, "dbscan": 0, "dpc": 0, "adjacency": 0, "shade": 0, "none": 0}
        for ds in best_offline[online_name]:
            ctr[best_offline[online_name][ds]] += 1

        if ctr["none"] > 0:
            online_name += "*"
        online_names.append(online_name)
        print(online_name, ctr)

        cumsum = 0
        plt.bar(x=i, height=ctr["adjacency"], hatch="o", align='center', alpha=1, color= offline_colors["adjacency"], edgecolor= "black")
        cumsum += ctr["adjacency"]
        plt.bar(x=i, height=ctr["kmeans"], bottom=cumsum, hatch="*", align='center', alpha=1, color= offline_colors["kmeans"], edgecolor= "black")
        cumsum += ctr["kmeans"]
        plt.bar(x=i, height=ctr["dbscan"], bottom=cumsum, hatch="x", align='center', alpha=1, color=offline_colors["dbscan"], edgecolor= "black")
        cumsum += ctr["dbscan"]
        plt.bar(x=i, height=ctr["dpc"], bottom=cumsum, hatch="/", align='center', alpha=1, color=offline_colors["dpc"], edgecolor= "black")
        cumsum += ctr["dpc"]
        plt.bar(x=i, height=ctr["shade"], bottom=cumsum, hatch="|", align='center', alpha=1, color=offline_colors["shade"], edgecolor= "black")

    plt.xticks(ticks=np.arange(len(main_online_algorithms)), labels=online_names, rotation=45, ha='right')
    plt.xlim([-0.5, len(main_online_algorithms)-0.5])
    plt.ylim([0, 19])
    plt.ylabel("Cumulative Count")
    #https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig("result_figures/offline_algorithms.pdf", bbox_inches='tight')

    plt.figure(figsize=(7, 1))

    plt.bar(0, 0, label="Adjacency", color=offline_colors["adjacency"], hatch="o", lw=1)
    plt.bar(0, 0, label="k-Means", color= offline_colors["kmeans"], hatch="*", lw=1)
    plt.bar(0, 0, label="DBSCAN", color=offline_colors["dbscan"], hatch="x", lw=1)
    plt.bar(0, 0, label="DPC", color=offline_colors["dpc"], hatch="/", lw=1)
    plt.bar(0, 0, label="SHADE", color=offline_colors["shade"], hatch="|", lw=1)
    plt.legend(framealpha=1.0, edgecolor="black", ncols=5, prop={'size': 29})
    plt.axis("off")
    plt.savefig("result_figures/offline_algorithms_legend.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    make_bar_plot()