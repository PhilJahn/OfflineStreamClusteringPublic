import numpy as np
from matplotlib import pyplot as plt
from correlation_extractor import get_scores_from_file
from defines import complete_real_datasets, online_method_names, offline_colors, dataset_names


def make_purity_plot(online_method):
    black = 'black'

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    dss = complete_real_datasets
    dss = np.sort(dss)

    plt.figure(figsize=(6, 2))

    online_purity = get_scores_from_file(f"result_extracts/scores_seed_online_main_real_completed_purity.txt")[online_method_names[online_method]]
    kmeans_purity = get_scores_from_file(f"result_extracts/scores_kmeans_main_real_completed_purity.txt")[online_method_names[online_method]]
    dbscan_purity = get_scores_from_file(f"result_extracts/scores_dbscan_main_real_completed_purity.txt")[online_method_names[online_method]]
    dpc_purity = get_scores_from_file(f"result_extracts/scores_dpc_main_real_completed_purity.txt")[online_method_names[online_method]]
    adjacency_purity = get_scores_from_file(f"result_extracts/scores_adjacency_main_real_completed_purity.txt")[online_method_names[online_method]]
    shade_purity = get_scores_from_file(f"result_extracts/scores_shade_main_real_completed_purity.txt")[online_method_names[online_method]]

    purity_val_list = []
    kmeans_val_list = []
    dbscan_val_list = []
    shade_val_list = []
    adjacency_val_list = []
    dpc_val_list = []

    ds_names = []
    for i in range(len(dss)):
        ds = dss[i]
        purity_val_list.append(online_purity[ds])
        kmeans_val_list.append(kmeans_purity[ds])
        dbscan_val_list.append(dbscan_purity[ds])
        shade_val_list.append(shade_purity[ds])
        adjacency_val_list.append(adjacency_purity[ds])
        dpc_val_list.append(dpc_purity[ds])
        ds_names.append(dataset_names[ds])
    #https://stackoverflow.com/questions/62683992/in-an-overlapping-scatter-plot-how-to-give-preference-to-a-specific-data
    plt.plot(range(len(dss)), kmeans_val_list, color=offline_colors["kmeans"], zorder=0)
    plt.plot(range(len(dss)), dbscan_val_list, color=offline_colors["dbscan"], zorder=0)
    plt.plot(range(len(dss)), shade_val_list, color=offline_colors["shade"], zorder=0)
    plt.plot(range(len(dss)), adjacency_val_list, color=offline_colors["adjacency"], zorder=0)
    plt.plot(range(len(dss)), dpc_val_list, color=offline_colors["dpc"], zorder=0)
    plt.plot(range(len(dss)), purity_val_list, color=offline_colors["online"], zorder=0)

    plt.scatter(range(len(dss)), adjacency_val_list, color=offline_colors["adjacency"], label="Adjacency", zorder=1, marker="o")
    plt.scatter(range(len(dss)), kmeans_val_list, color=offline_colors["kmeans"], label="k-Means", zorder=1, marker='*')
    plt.scatter(range(len(dss)), dbscan_val_list, color=offline_colors["dbscan"], label="DBSCAN", zorder=1, marker='x')
    plt.scatter(range(len(dss)), dpc_val_list, color=offline_colors["dpc"], label="DPC", zorder=1, marker='+')
    plt.scatter(range(len(dss)), shade_val_list, color=offline_colors["shade"], label="SHADE", zorder=1, marker='s')
    plt.scatter(range(len(dss)), purity_val_list, color=offline_colors["online"], label="Online", zorder=1, marker='d')

    plt.xticks(range(len(dss)), ds_names)
    plt.ylabel("Purity")
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1, 0, 0.3, 1))

    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"result_figures/purity_{online_method}.pdf", bbox_inches='tight', transparent=True)
    plt.show()





if __name__ == "__main__":
    online_methods = ["clustream_river", "denstream_river", "batch_kcentroid", "window_kcentroid", "stmstream", "swclustering", "dstream", "birch_sklearn"]
    for online_method in online_methods:
        make_purity_plot(online_method)