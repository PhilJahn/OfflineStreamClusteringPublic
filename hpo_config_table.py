import math
import traceback

import numpy as np

from datahandler import load_data
from defines import full_online_algorithms, real_datasets, online_method_names

if __name__ == "__main__":
    ds_batch_nums = {}
    for ds in real_datasets:
        data_points, labels = load_data(ds, 0)
        batch_num = int(np.ceil(len(data_points) / 1000))
        ds_batch_nums[ds] = batch_num

    table_content = ""

    for online in full_online_algorithms:
        if online not in ["full_batch", "pure_offline"]:
            final_count_dss = []
            for ds in real_datasets:
                final_count_ds = 0
                for s in [0,1,2]:
                    with open(f"online_param_logs/{ds}/online_params_{ds}_{online}_b1000_e1_t300_logdim_s{s}.csv", "r") as f:
                        lines = f.readlines()
                        final_count = lines[1].split(";")[3]
                        final_count_ds += int(final_count)
                final_count_ds /= 3.0
                final_count_dss.append(final_count_ds)
            final_count_all = np.mean(final_count_dss)
            table_content += f"{online_method_names[online]} & {final_count_all:.1f}"
        else:
            table_content += f"{online_method_names[online]} & -"


        for offline in ["kmeans", "dbscan", "dpc"]:
            final_count_dss = []
            if (online == "dstream" or online == "denstream_river") and offline == "dbscan":
                offline = "fdbscan"
            for ds in real_datasets:
                if online != "pure_offline":
                    final_counts_ds = []
                    for batch_num in range(ds_batch_nums[ds]):
                        try:
                            final_count_batch = 0
                            for s in [0, 1, 2]:
                                with open(f"offline_param_logs/{ds}/{online}/{offline}/offline_params_{ds}_{online}_{offline}_b1000_t1300_logdim_t2600_logdim_c{batch_num}_o{s}.csv",
                                          "r") as f:
                                    lines = f.readlines()
                                    final_count = lines[1].split(";")[2]
                                    final_count_batch += int(final_count)
                            final_count_batch /= 3.0
                            final_counts_ds.append(final_count_batch)
                        except Exception as e:
                            #print(traceback.format_exc(), flush=True)
                            print(f"Could not find {online}  + {offline} for {ds}")
                    final_count_ds = np.mean(final_counts_ds)
                else:
                    final_count_ds = 0
                    for s in [0, 1, 2]:
                        try:
                            with open(
                                    f"pure_offline_param_logs/{ds}/pure_offline_params_{ds}_pure_{offline}_b1000_e1_t900_logdim_s{s}.csv",
                                    "r") as f:
                                lines = f.readlines()
                                final_count = lines[1].split(";")[3]
                                final_count_ds += int(final_count)
                        except:
                            #print(traceback.format_exc(), flush=True)
                            print(f"Could not find {online}  + {offline} for {ds}")
                            final_count_ds = 0
                    final_count_ds /= 3.0
                if not math.isnan(final_count_ds):
                    final_count_dss.append(final_count_ds)
            print(online, offline, final_count_dss)
            final_count_all = np.mean(final_count_dss)
            table_content += f"& {final_count_all:.1f}"

        table_content += "\\\\\n"

    with open("table_hpo_configs.txt", "w") as f:
        f.write(table_content)