import copy

import utils
from clustream import CluStream
import numpy as np
from datahandler import load_data
from online_handler import get_online_method


def process_summary(online_alg, summary, batch_datapoints, batch_start_id, do_pseudoclustering):
    if online_alg in ["clustream", "denstream", "hddstream", "streamkmeans", "streamkmplus"]:
        if online_alg == "clustream":
            positions = np.array([mc.mean for mc in summary.values()])
            radii = np.array([mc.boundary for mc in summary.values()])
            weights = np.array([mc.n for mc in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "denstream" or online_alg == "hddstream":
            pmcs = summary["pmcs"]  # DenStream discards omcs for clustering
            mu = pmcs[list(pmcs.keys())[0]].mu
            positions = [pmc.mean for pmc in pmcs.values()]
            radii = [pmc.radius for pmc in pmcs.values()]
            weights = [pmc.w for pmc in pmcs.values()]
            id_map = list(pmcs.keys())
            status = np.where(np.array(weights) >= mu, "dense", "normal").tolist()

            omcs = summary["omcs"]
            positions.extend([omc.mean for omc in omcs.values()])
            radii.extend([omc.radius for omc in omcs.values()])
            weights.extend([omc.w for omc in omcs.values()])
            id_map.extend(list(omcs.keys()))
            status.extend(["outlie"] * len(omcs.values()))

            positions = np.array(positions)
            radii = np.array(radii)
            weights = np.array(weights)
            id_map = np.array(id_map)
            status = np.array(status)
        elif online_alg == "streamkmeans":
            positions = np.array([prototype.center for prototype in summary.values()])
            radii = np.zeros(len(summary))
            weights = np.array([prototype.weight for prototype in summary.values()])
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "streamkmplus":
            positions = np.array(list(summary.values()))
            radii = np.zeros(len(summary))
            weights = np.ones(len(summary))
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
        #print(utils.distance_calc(batch_datapoints, positions, dist="euclidean").shape)
        assignment_dists = utils.distance_calc(batch_datapoints, positions, dist="euclidean")
        assignment = np.argmin(assignment_dists, axis=1)
    elif online_alg == "dstream":
        ex_grid_cell = summary[list(summary.keys())[0]]
        radius = (ex_grid_cell.upper_bound[0] - ex_grid_cell.lower_bound[0]) / 2 + 0.000001 # acount for roudning error
        #print(radius)
        cm = ex_grid_cell.cm
        cl = ex_grid_cell.cl
        grid_num = ex_grid_cell.grid_num
        l = ex_grid_cell.l
        divisor = (grid_num * (1 - l))
        dense = cm / divisor
        normal = cl / divisor
        d = len(batch_datapoints[0])
        upper_bounds = np.array([cell.upper_bound for cell in summary.values()])
        lower_bounds = np.array([cell.lower_bound for cell in summary.values()])
        positions = ((upper_bounds + lower_bounds) / 2).tolist()
        outlier_pos = np.array([np.inf]*d)
        positions.append(outlier_pos)
        positions = np.array(positions)
        #print(positions)
        radii = [radius] * len(summary)
        radii.append(0)
        radii = np.array(radii)
        weights = [cell.density for cell in summary.values()]
        weights.append(0)
        weights = np.array(weights)

        id_map = list(summary.keys())
        id_map.append("deleted")
        id_map = np.array(id_map)

        status = np.where(np.array(weights) >= dense, "dense", "normal")
       # print(status)
        status[np.where(np.array(weights) < normal)] = "outlie"
        #print(status)
        status[np.where(np.array([cell.status for cell in summary.values()]) == "sporadic")] = "sporad"
        #print(status)
        status[-1] = "deleted"
        # print(utils.distance_calc(batch_datapoints, positions, dist="euclidean").shape)
        # assignment_dists = utils.distance_calc(batch_datapoints, positions, dist="euclidean")
        # assignment = np.argmin(assignment_dists, axis=1)
        grid_coordinates = batch_datapoints // (2*radius)
        #print("grid_coordinates", grid_coordinates)
        assignment = []
        summary_key_list = list(summary.keys())
        for grid_coordinate in grid_coordinates:
            cell_id_list = grid_coordinate.tolist()
            #print(cell_id_list)
            cell_id = str(cell_id_list).replace("[", "").replace("]", "").replace(".0", "").replace(" ", "")
            if cell_id in summary_key_list:
                assignment.append(summary_key_list.index(cell_id))
            else:
                assignment.append(-1)
        assignment = np.array(assignment)
       # print(assignment)
    elif online_alg == "emcstream":
        assignment = np.arange(len(batch_datapoints))  # self assignment
        id_map = assignment
        radii = np.zeros(len(batch_datapoints))
        weights = np.ones(len(batch_datapoints))
        positions = summary["embedding"]
        status = np.array(["dense"] * len(weights))
    elif online_alg == "gbfuzzystream" or online_alg == "stmstream":
        if online_alg == "gbfuzzystream":
            positions = np.array([fmb.center for fmb in summary.values()])
            radii = np.array([fmb.radius for fmb in summary.values()])
            weights = np.array([len(fmb.samples) for fmb in summary.values()])
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
            sample_ids = [fmb.sample_ids for fmb in summary.values()]
        elif online_alg == "stmstream":
            positions = np.array([cell.nucleus for cell in summary.values()])
            radii = np.array([cell.radius for cell in summary.values()])
            weights = np.array([len(cell.samples) for cell in summary.values()])
            id_map = np.array(list(summary.keys()))
            noise = np.array([cell.noise for cell in summary.values()])
            aged_out = np.array([cell.aged_out for cell in summary.values()])
            noisy = np.logical_or(aged_out, noise)
            status = np.where(noisy, "outlie", "dense").tolist()
            sample_ids = [cell.sample_ids for cell in summary.values()]
        assignment = []
        for j in range(len(batch_datapoints)):
            j_id = j + batch_start_id
            found = False
            for h in range(len(sample_ids)):
                if j_id in sample_ids[h]:
                    assignment.append(h)
                    found = True
                    break
            if not found:  # missing
                print(f"{j_id} not found in summary")
                assignment.append(np.argmin(utils.distance_calc(batch_datapoints[j], positions, dist="euclidean")))
            assert len(assignment) == j + 1
        assignment = np.array(assignment)
        print(len(assignment))
    else:
        raise NotImplementedError(f"{online_alg} is not implemented.")
    min_store = {"id_map": id_map, "center": positions, "radii": radii, "weights": weights, "status": status}
    if do_pseudoclustering:
        if online_alg in ["clustream", "denstream", "hddstream", "dstream", "gbfuzzystream",
                          "stmstream"]:  # connected components
            mc_dists = utils.distance_calc(positions, positions, dist="euclidean")
            two_radii = np.array([radii + radii[i] for i in range(len(positions))])
            connected = np.where(mc_dists <= two_radii, True, False)
            partitions = np.arange(len(positions))  # get connected
            for i in range(len(partitions)):
                if status[i] != "outlie":
                    for j in range(i):
                        if status[j] != "outlie":
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

            return assignment, min_store, partitions[assignment]
        elif online_alg == "emcstream":
            psuedoclustering = summary["labels"]
            return assignment, min_store, np.array(psuedoclustering)
        else:  # no pseudo-clustering
            psuedoclustering = assignment
            return assignment, min_store, psuedoclustering
    else:
        return assignment, min_store


import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from clustpy.metrics.clustering_metrics import purity

if __name__ == "__main__":
    stream, labels = load_data("complex9")
    online_alg = "emcstream"
    stream_alg = get_online_method(online_alg, {"dim_grid_num":10}, 0)
    batch_size = 1000
    batch_ids = np.arange(0, len(stream), step=batch_size).tolist()
    batch_ids.append(len(stream))
    print(batch_ids)
    timestamps = np.arange(len(stream))
    if online_alg == "emcstream":  # needs to run through fully
        stream_alg.learn(stream.tolist(), timestamps.tolist())
        emc_summary = stream_alg.get_summary()
    summary = None
    prior_summary = None
    for i in range(len(batch_ids) - 1):
        print(i, len(stream[batch_ids[i]:batch_ids[i + 1]]))
        stream_batch = stream[batch_ids[i]:batch_ids[i + 1]]
        label_batch = labels[batch_ids[i]:batch_ids[i + 1]]
        timestamp_batch = timestamps[batch_ids[i]:batch_ids[i + 1]]
        #print(timestamp_batch)
        if online_alg != "emcstream":
            _, event_log = stream_alg.learn(stream_batch.tolist(), timestamp_batch.tolist())
            print(event_log)
            summary = stream_alg.get_summary()
            if online_alg == "clustream":
                summary_tmp = copy.deepcopy(summary)
                removal_ids = []
                if prior_summary is not None:
                    for mc_old in prior_summary.values():
                        for mc_new in summary.values():
                            if mc_old.mc_id in mc_new.mc_history:
                                print("subtract", mc_old.mc_id, "from", mc_new.mc_id)
                                print(mc_new, mc_old)
                                mc_new.subtract(mc_old)
                                if mc_new.count == 0: # empty MC after subtraction
                                    removal_ids.append(mc_new.mc_id)
                                print(mc_new)
                    for del_mc_id in removal_ids:
                        summary.pop(del_mc_id)

                    centers = np.array([mc.mean for mc in summary.values()])
                    centers_ones = np
                    dist = utils.distance_calc(centers, centers, dist="euclidean")
                    mc_ids = list(summary.keys())
                    for mc_index in range(len(mc_ids)):
                        mc_id = mc_ids[mc_index]
                        mc = summary[mc_id]
                        if mc.count == 1:
                            mc.boundary = np.partition(dist[mc_index],1)[1]
                prior_summary = copy.deepcopy(summary_tmp)
        else:
            summary = {"embedding": emc_summary["embedding"][batch_ids[i]:batch_ids[i + 1]],
                       "timestamps": emc_summary["timestamps"][batch_ids[i]:batch_ids[i + 1]],
                       "labels": emc_summary["labels"][batch_ids[i]:batch_ids[i + 1]]}
        #(summary[list(summary.keys())[-1]])
        print(summary)
        np.save(f"./storage_test/{online_alg}_store_{i}.npy", summary)
        # https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
        summary2 = np.load(f"./storage_test/{online_alg}_store_{i}.npy", allow_pickle=True)
        summary2 = summary2.item()
        #print(summary2[list(summary2.keys())[-1]])
        assignment, min_store, psuedoclustering = process_summary(online_alg, summary2, stream_batch, batch_ids[i],
                                                                  True)
        _, psuedoclustering = np.unique(psuedoclustering, return_inverse=True)
        plt.figure(figsize=(8, 8))
        print(min_store)
        ids = min_store["id_map"]
        centers = min_store["center"]
        weights = min_store["weights"]
        radii = min_store["radii"]
        status = min_store["status"]
        for l in range(len(ids)):
            # print(mc.mean)
            mc_patch = ptc.Circle((float(centers[l][0]), float(centers[l][1])), radii[l], alpha=0.4, color="lightgrey")
            plt.gca().add_patch(mc_patch)
            if status[l] == "dense":
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l] * 10, c="darkgrey", marker="X", linewidths=0)
            elif status[l] == "normal":
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=weights[l] * 10, c="darkgrey", marker="s")
            elif status[l] == "outlie":
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=max(1,weights[l]) * 15, c="darkgrey", marker="+")
            else:
                plt.scatter(x=centers[l][0], y=centers[l][1], alpha=1, s=max(1,weights[l]) * 15, c="darkgrey", marker="*", linewidths=0)
            if online_alg != "dstream":
                plt.text(centers[l][0], centers[l][1], s=ids[l])

        plt.scatter(x=stream_batch[:, 0], y=stream_batch[:, 1], cmap=plt.get_cmap("jet"), c=psuedoclustering, s=10)
        for j in range(stream_batch.shape[0]):
            plt.text(stream_batch[j][0], stream_batch[j][1], s=psuedoclustering[j], c="grey")
        assignpurity = purity(labels_true=label_batch, labels_pred=assignment)
        pseudopurity = purity(labels_true=label_batch, labels_pred=psuedoclustering)
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.title(f"assign purity: {assignpurity}; pseudo purity: {pseudopurity}")
        plt.savefig(f"./test_figures_2/{online_alg}_store_{i}.png")
        plt.close()
