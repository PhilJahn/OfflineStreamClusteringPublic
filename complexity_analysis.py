import argparse

import matplotlib.pyplot as plt
import numpy as np

from defines import online_method_names, online_colors, online_linestyles, main_online_algorithms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling', default="k", type=str, help='Scaling component')
    parser.add_argument('--resource', default="time", type=str, help='Resource to plot')
    args = parser.parse_args()

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 28})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    online_list = main_online_algorithms
    birch_index = online_list.index("birch_sklearn")
    online_list[birch_index] = "birch_sklearn_2"

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.append(colors, [[0, 0, 0, 1]], axis=0)

    seeds_1 = [0,1,2,3,4]
    seeds_2 = [0,1,2,3,4]
    if args.scaling == "n":
        ns = [5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000]
        ds = [2]
        ks = [100]
    elif args.scaling == "k":
        ns = [50000]
        ds = [2]
        ks = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    elif args.scaling == "d":
        ns = [50000]
        ds = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
        ks = [100]

    results_mean = {}
    results_std = {}

    plt.figure(figsize=(10,4))
    i = 0
    for online in online_list:
        results_mean[online] = []
        results_std[online] = []
        for d in ds:
            for k in ks:
                for n in ns:
                    result_values = []
                    for seed2 in seeds_2:
                        complete = True
                        for seed1 in seeds_1:
                            try:
                                with open(f"resource_eval/{online}_n{n}_d{d}_k{k}_f{seed1}_s{seed2}.txt", "r") as f:
                                    content = f.readlines()[1]
                                    content_split = content.split(",")
                                    if args.resource == "time":
                                        result_values.append(float(content_split[3]))
                                    elif args.resource == "full_memory":
                                        result_values.append(float(content_split[0]) + float(content_split[1]))
                                    elif args.resource == "full_memory_2":
                                        result_values.append(float(content_split[0]) + float(content_split[2]))
                                    elif args.resource == "summary":
                                        result_values.append(float(content_split[2]))
                            except:
                                print(f'sbatch --output="./process_log/resource_eval/{online}_n{n}_d{d}_k{k}_{seed1}_{seed2}.txt" ./runResourceEval.sh {online} {n} {d} {k} {seed1} {seed2}')
                    if len(result_values) == 25:
                        results_mean[online].append(np.mean(result_values))
                        results_std[online].append(np.std(result_values))
                    else:
                        results_mean[online].append(np.nan)
                        results_std[online].append(np.nan)
        results_mean[online] = np.array(results_mean[online])
        results_std[online] = np.array(results_std[online])
        if args.scaling == "n":
            iterated_resources = np.array(ns)
        elif args.scaling == "k":
            iterated_resources = np.array(ks)
        elif args.scaling == "d":
            iterated_resources = np.array(ds)

        plt.fill_between(iterated_resources, results_mean[online] - results_std[online], results_mean[online] + results_std[online], alpha=0.2, color=online_colors[online])
        plt.plot(iterated_resources, results_mean[online], label=online_method_names[online], color=online_colors[online], linestyle=online_linestyles[online], lw=3.5)
        i += 1
    plt.xscale("log")
    plt.yscale("log")
    if args.scaling == "n":
        plt.xlabel("Number of Instances")
    elif args.scaling == "d":
        plt.xlabel("Dimensionality")

    if args.resource == "time":
        plt.ylabel("Time (in Seconds)")
    elif args.resource == "full_memory":
        plt.ylabel("Memory (in Bytes)")
    plt.savefig(f"result_figures/scaling_{args.scaling}_{args.resource}.pdf", bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 2))
    for key in main_online_algorithms:
        plt.plot(0, 1000, label=online_method_names[key], color=online_colors[key], ls=online_linestyles[key], lw=2)
    plt.legend(framealpha=1.0, edgecolor="black", ncols=4)
    plt.axis("off")
    plt.savefig("result_figures/scaling_legend.pdf", bbox_inches='tight')
    plt.show()


