import os
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from defines import online_linestyles, online_colors, all_datasets, main_online_algorithms, select_online_algorithms, \
    online_method_names


def make_drift_figure(ds, metric, online_list, offline):

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    result_store = {}
    result_store_std = {}
    result_store[metric] = {}
    result_store_std[metric] = {}
    for online in online_list:
        if offline != "online":
            suffix = "1000_300_logdim_600_logdim"
            store_name = f"result_evals/{offline}/{ds}_{online}_{offline}_{suffix}.npy"
            std_name = f"result_evals/{offline}/{ds}_{online}_{offline}_{suffix}_std.npy"
        else:
            suffix = "1000_300_logdim"
            store_name = f"result_evals/{offline}/{ds}_{online}_{suffix}.npy"
            std_name = f"result_evals/{offline}/{ds}_{online}_{suffix}_std.npy"

        if os.path.exists(store_name):
            results = np.load(store_name, allow_pickle=True).item()
            results_std = np.load(std_name, allow_pickle=True).item()
            result_store[metric][online] = []
            result_store_std[metric][online] = []
            for timestamp in results.keys():
                if timestamp != "all" and timestamp != "all_std":
                    result_store[metric][online].append(results[timestamp][metric])
                    result_store_std[metric][online].append(results_std[timestamp][metric])
        else:
            print(store_name, "is missing")

    print(result_store[metric])

    if ds != "sensorstream" and ds != "covertype" and ds != "kddcup99":
        plt.figure(figsize = (8,2))
    else:
        plt.figure(figsize = (16,2))

    if ds == "iabrupt":
        plt.vlines(14352//1000, 0, 0.5, color = "black")
        plt.vlines(19500//1000, 0, 0.5, color = "black")
        plt.vlines(33240//1000, 0, 0.5, color = "black")
        plt.vlines(38682//1000, 0, 0.5, color="black")
        plt.vlines(39510//1000, 0, 0.5, color="black")
    elif ds == "igradual":
        plt.vlines(14028//1000, 0, 0.5, color="black")
    elif ds == "ireocurring" or ds == "iabruptreocurring":
        plt.vlines(26568//1000, 0, 0.5, color="black")
        plt.vlines(53364//1000, 0, 0.5, color="black")

    plt.xlabel("Batch Number")
    plt.ylabel("TS Index")


    for online in online_list:
        try:
            plt.fill_between(np.arange(len(result_store[metric][online])), np.array(result_store[metric][online]) - np.array(result_store_std[metric][online]),np.array(result_store[metric][online]) + np.array(result_store_std[metric][online]), color=online_colors[online], alpha=0.25)
            plt.plot(result_store[metric][online], label=online, color=online_colors[online], linestyle=online_linestyles[online], lw=3)
        except:
            print(online, "is missing")
    #plt.legend()
    if not os.path.exists("drift_figures/"):
        os.makedirs("drift_figures/", exist_ok = True)


    plt.savefig(f"drift_figures/{ds}_{metric}.pdf", bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    drift_datasets = ["iabrupt", "igradual", "ireocurring", "iabruptreocurring", "iincremental", "covertype", "sensorstream", "stream_densired", "rbf3", "kddcup99"]
    select_online_algorithms = ["batch_kcentroid",
                                "window_kcentroid",
                                "streamkmeans_river", "clustream_river", "denstream_river",
                                "dstream", "gbfuzzystream_original", "stmstream"]
    for dataset in drift_datasets:
        make_drift_figure(dataset, "batch_ts", select_online_algorithms, "online")

    plt.figure(figsize=(10, 2))

    for key in select_online_algorithms:
        plt.plot(0, 1000, label=online_method_names[key], color=online_colors[key], ls=online_linestyles[key], lw=2.5)
    plt.legend(framealpha=1.0, edgecolor="black", ncols=4)
    plt.axis("off")
    plt.savefig("drift_figures/drift_legend.pdf", bbox_inches='tight')
    plt.show()