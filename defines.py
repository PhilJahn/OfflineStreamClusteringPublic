from copy import copy

online_method_names = {"full_batch": "Full Batch",
                       "batch_random": "Random (Batch)",
                       "batch_lwc": "LWC-M (Batch)",
                       "batch_lwc_square": "LWC (Batch)",
                       "batch_kcentroid": "k-Centroid (Batch)",
                       "batch_protras": "ProTraS (Batch)",
                       "window_random": "Random (Window)",
                       "window_lwc": "LWC-M (Window)",
                       "window_lwc_square": "LWC (Window)",
                       "window_kcentroid": "k-Centroid (Window)",
                       "window_protras": "ProTraS (Window)",
                       "streamkmeans_river": "StreamKMeans",
                       "streamkmplus": "StreamKM++",
                       "birch_sklearn": "BIRCH",
                       "birch_sklearn_2": "BIRCH",
                       "clustream_river": "CluStream",
                       "denstream_river": "DenStream",
                       "dstream": "D-Stream",
                       "swclustering": "SWClustering",
                       "gbfuzzystream_original": "GB-FuzzyStream",
                       "stmstream": "STM-Stream",
                       "emcstream_original": "EmCStream",
                       "pure_offline": "Full Data"
                       }

online_colors = {"full_batch": "lightgray",
                 "batch_random": "red",
                 "batch_lwc_square": "darkred",
                 "batch_kcentroid": "orangered",
                 "batch_protras": "coral",
                 "window_random": "green",
                 "window_lwc_square": "darkgreen",
                 "window_kcentroid": "lime",
                 "window_protras": "palegreen",
                 "streamkmeans_river": "tan",
                 "streamkmplus": "orange",
                 "birch_sklearn": "blue",
                 "birch_sklearn_2": "blue",
                 "clustream_river": "darkblue",
                 "denstream_river": "lightblue",
                 "swclustering": "aqua",
                 "dstream": "brown",
                 "gbfuzzystream_original": "deeppink",
                 "stmstream": "pink",
                 "emcstream_original": "darkviolet",
                 "pure_offline": "darkgray"
                 }

online_linestyles = {"full_batch": (0, ()),
                    "batch_random": (0, (5, 1)),
                    "batch_lwc_square": (0, (5, 1)),
                    "batch_kcentroid": (0, (5, 1)),
                    "batch_protras":(0, (5, 1)),
                    "window_random":  (0, (5, 5)),
                    "window_lwc_square": (0, (5, 5)),
                    "window_kcentroid": (0, (5, 5)),
                    "window_protras": (0, (5, 5)),
                    "streamkmeans_river": (0, (5, 10)),
                    "streamkmplus": (0, (5, 10)),
                    "birch_sklearn": (0, (1, 3)),
                    "birch_sklearn_2": (0, (1, 3)),
                    "clustream_river": (0, (1, 3)),
                    "denstream_river": (0, (1, 3)),
                    "swclustering": (0, (1, 3)),
                    "dstream": (0, (3, 5, 1, 5)),
                    "gbfuzzystream_original": (0, (1, 1)),
                    "stmstream": (0, (1, 1)),
                    "emcstream_original": (0, (1, 1)),
                    "pure_offline": (0, ())
                    }

dataset_names = {"stream_densired": "SDENS",
                 "rbf3": "RBF-3",
                 "har": "HAR",
                 "electricity": "Elec",
                 "iabrupt": "InsA",
                 "iincremental": "InsI",
                 "gassensor": "GasS",
                 "pendigits": "PenD",
                 "starlightcurves": "SLC",
                 "adult": "Adult",
                 "ireocurring": "InsR",
                 "igradual": "InsG",
                 "iabruptreocurring": "InsAR",
                 "posture": "Post",
                 "smartmeter": "Smart",
                 "uwave": "UWave",
                 "asfault": "Asflt",
                 "powersupply": "Power",
                 "outdoor": "OdO",
                 "htru2": "HT2",
                 "cancer": "Canc",
                 "sepsissurvival": "SSMCR",
                 "mirs": "MIRS",
                 "yoga": "Yoga",
                 "noaa": "NOAA",
                 "kddcup99": "KDD99",
                 "covertype": "CovT",
                 "sensorstream": "Sensor",
                 "pokerhand": "Poker",
                 "heartbeats": "HeaBea",
                 "rialto": "RiaBT",
                 "phiusiil": "PPURL",
                 "rank": "Avg. Rank"}

main_online_algorithms = ["batch_random", "batch_lwc_square", "batch_kcentroid", "batch_protras",
                           "window_random", "window_lwc_square", "window_kcentroid", "window_protras",
                           "streamkmeans_river", "streamkmplus", "birch_sklearn", "clustream_river", "denstream_river", "swclustering",
                           "dstream", "gbfuzzystream_original", "stmstream"]

select_online_algorithms = ["batch_random", "batch_kcentroid",
                           "window_random", "window_kcentroid",
                           "streamkmeans_river", "birch_sklearn", "clustream_river", "denstream_river",
                           "dstream", "gbfuzzystream_original", "stmstream"]

full_online_algorithms = ["full_batch", "batch_random", "batch_lwc_square", "batch_kcentroid", "batch_protras",
                           "window_random", "window_lwc_square", "window_kcentroid", "window_protras",
                           "streamkmeans_river", "streamkmplus", "birch_sklearn", "clustream_river", "denstream_river", "swclustering",
                           "dstream", "gbfuzzystream_original", "stmstream"]

all_online_algorithms = ["full_batch", "batch_random", "batch_lwc_square", "batch_kcentroid", "batch_protras",
                           "window_random", "window_lwc_square", "window_kcentroid", "window_protras",
                           "streamkmeans_river", "streamkmplus", "birch_sklearn", "clustream_river", "denstream_river", "swclustering",
                           "dstream", "emcstream_original", "gbfuzzystream_original", "stmstream"]

real_datasets = ["kddcup99", "covertype", "sensorstream", "pokerhand", "heartbeats", "rialto", "phiusiil", "starlightcurves", "har", "pendigits", "adult", "iabrupt", "iincremental", "ireocurring", "igradual", "iabruptreocurring", "posture", "smartmeter", "uwave", "asfault", "powersupply", "outdoor", "gassensor", "htru2", "cancer", "sepsissurvival", "electricity", "mirs", "yoga", "noaa"]
synth_datasets = ["stream_densired", "rbf3"]

all_datasets = copy(synth_datasets)
all_datasets.extend(real_datasets)

complete_real_datasets = ["covertype", "pokerhand", "har", "pendigits", "adult", "iabrupt", "iincremental", "ireocurring", "igradual", "iabruptreocurring", "posture", "uwave", "asfault", "outdoor", "gassensor", "htru2", "yoga", "noaa", "electricity"]

smaller_better_metrics = ["davies_bouldin_score", "offline_score", "online_score"]

metric_names = {"ari": "ARI", "accuracy": "Accuracy", "ami": "AMI", "purity": "Purity", "batch_ts": "TS",
                      "offline_score": "ARI+AMI", "clu_num": "q",
                      "cmm": "CMM", "cmm_alt": "CMM+", "fowlkes_mallows": "FMI", "nmi": "NMI",
                      "homogeneity": "HOM", "completeness": "COM", "fair_nmi": "FNMI", "variation_of_information": "VI",
                      "davies_bouldin_score": "DBI", "calinski_harabasz_score": "CHI", "disco": "DISCO",
                        "simplified_silhouette": "SSIL", "summary_ssil": "SSIL",
                        "bop_score_real": "BoP$_{R}$", "bop_score_summary": "BoP$_{S}$", "bop_score": "BOP",
                        "summary_chi": "CHI", "adjacency": "Adjacency"
                      }

corr_metric_names = {"purity": "Purity", "batch_ts": "TS",
                      "offline_score": "Online\n(ARI+AMI)", "clu_num": "SS Count", "summary_ssil": "SSIL",
                        "bop_score_real": "BoP$_{R}$", "bop_score_summary": "BoP$_{S}$", "bop_score": "BOP",
                        "summary_chi": "CHI", "adjacency": "Adjacency\n(ARI+AMI)"
                      }

offline_colors = {"dbscan": "cyan", "dpc": "lavender", "kmeans": "silver", "adjacency": "gold", "shade": "orange", "online": "slategray"}

if __name__ == "__main__":
    print(len(all_datasets))
    print(len(synth_datasets))
    print(len(complete_real_datasets))