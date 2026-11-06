from matplotlib import pyplot as plt

from defines import metric_names, corr_metric_names


def make_correlation_boxplot(online_metrics, offline_method):



    black = 'black'

    # https://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.figure(figsize=(6, 3))
    for i in range(len(online_metrics)):
        online_metric = online_metrics[i]
        with open(f"corr_scores/{online_metric}_to_{offline_method}.txt", "r") as f:
            corrs = []
            pvalues = []
            content = f.readlines()
            print(content)
            stat_signs = {"-": 0, "+": 0}
            for line in content:
                linesplit = line.replace("\n", "").split(",")
                corr = float(linesplit[1])
                pvalue = float(linesplit[2])
                corrs.append(corr)
                pvalues.append(pvalue)

                if pvalue <= 0.05:
                    if corr < 0:
                        stat_signs["-"] += 1
                    else:
                        stat_signs["+"] += 1

            if online_metric in ["purity", "offline_score", "adjacency"]:
                facecolor = "lightgrey"
            elif online_metric in ["bop_score_real", "bop_score_summary", "batch_ts"]:
                facecolor = "lightblue"
            elif online_metric in ["summary_ssil", "summary_chi"]:
                facecolor = "salmon"
            elif online_metric in ["clu_num"]:
                facecolor = "lightgreen"

            plt.boxplot(pvalues, tick_labels=[corr_metric_names[online_metric]], positions=[i], widths=[0.5], patch_artist=True,
                        boxprops=dict(facecolor=facecolor, color=black),
                        capprops=dict(color=black),
                        whiskerprops=dict(color=black),
                        flierprops=dict(color=black, markeredgecolor=black),
                        medianprops=dict(color=black))
            if stat_signs["-"] == 0 and stat_signs["+"] == 0:
                plt.text(y=2, x=i, s="-", ha="center", va="center")
            else:
                plt.text(y=2, x=i, s=f"{stat_signs['+']/(stat_signs['-']+stat_signs['+']):.2f}", ha="center", va="center")
    plt.hlines(y=0.05, xmin=-0.5, xmax=len(online_metrics)-0.5, linestyles="dashed", color="red")
    plt.xticks(rotation=45, ha='right')
    plt.xlim(-0.5, len(online_metrics)-0.5)
    plt.ylim(1e-4, 5)
    plt.yscale("log")
    plt.ylabel("p-Value")
    #plt.xlabel("Online Quality Metric")
    plt.savefig(f"result_figures/correlation_boxplot_{offline_method}.pdf", bbox_inches='tight', transparent=True)
    plt.close()
    plt.show()

if __name__ == "__main__":
    offline_methods = ["dpc", "dbscan", "kmeans", "adjacency", "shade"]
    online_metrics = ["offline_score", "purity", "adjacency", "summary_ssil", "summary_chi", "batch_ts", "bop_score_real", "bop_score_summary", "clu_num"]

    for offline_method in offline_methods:
        make_correlation_boxplot(online_metrics, offline_method)