import copy

import utils
from clustream import CluStream
import numpy as np
from datahandler import load_data
from online_handler import get_online_method


def process_summary(online_alg, summary, batch_datapoints, batch_start_id, do_pseudoclustering):
    if online_alg in ["clustream", "denstream", "hddstream", "streamkmeans", "streamkmeans+", "streamkmplus",
                      "streamkmplus+", "birch", "swclustering", "swclustering+",
                      "clustream_river", "denstream_river", "streamkmeans_river", "birch_sklearn"]:
        if online_alg == "clustream":
            positions = np.array([mc.mean for mc in summary.values()])
            radii = np.array([mc.boundary for mc in summary.values()])
            weights = np.array([mc.n for mc in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "denstream" or online_alg == "hddstream":
            pmcs = summary["pmcs"]  # DenStream discards omcs for clustering
            positions = [pmc.mean for pmc in pmcs.values()]
            radii = [pmc.radius for pmc in pmcs.values()]
            weights = [pmc.w for pmc in pmcs.values()]
            if len(pmcs.keys()) > 0:
                mu = pmcs[list(pmcs.keys())[0]].mu
                status = np.where(np.array(weights) >= mu, "dense", "normal").tolist()
            else:
                status = []
            id_map = list(pmcs.keys())


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
        elif online_alg == "streamkmeans" or online_alg == "streamkmeans+":
            positions = np.array([prototype.center for prototype in summary.values()])
            radii = np.zeros(len(summary))
            weights = np.array([prototype.weight for prototype in summary.values()])
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "streamkmplus" or online_alg == "streamkmplus+":
            coreset_points = list(summary.values())
            positions = np.array([coreset_point.position for coreset_point in coreset_points])
            radii = np.zeros(len(summary))
            weights = np.array([coreset_point.weight for coreset_point in coreset_points])
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "birch":
            entries = summary["entries"]
            positions = [entry.mean for entry in entries]
            radii = [entry.radius for entry in entries]
            weights = [entry.n for entry in entries]
            id_map = np.arange(len(entries)).tolist()
            status = np.array(["dense"] * len(weights)).tolist()

            outliers = summary["outliers"]
            positions.extend([outlier.mean for outlier in outliers])
            radii.extend([outlier.radius for outlier in outliers])
            weights.extend([outlier.n for outlier in outliers])
            id_map.extend((np.arange(len(outliers))+len(id_map)).tolist())
            status.extend(["outlie"] * len(outliers))

            positions = np.array(positions)
            radii = np.array(radii)
            weights = np.array(weights)
            id_map = np.array(id_map)
            status = np.array(status)
        elif online_alg == "swclustering" or online_alg == "swclustering+":
            positions = np.array([ehcf.mean for ehcf in summary.values()])
            radii = np.array([ehcf.boundary for ehcf in summary.values()])
            weights = np.array([ehcf.n for ehcf in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "clustream_river":
            r_factor = list(summary.values())[0].r_factor
            positions = utils.dps_to_np([mc.center for mc in summary.values()])
            radii = np.array([mc.radius(r_factor) for mc in summary.values()])
            weights = np.array([mc.weight for mc in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "denstream_river":

            pmcs = summary["pmcs"]  # DenStream discards omcs for clustering

            if len(pmcs.keys()) > 0:
                summary_timestamp = list(pmcs.values())[0].summary_timestamp
                mu = list(pmcs.values())[0].mu
                #print(summary_timestamp, mu)
                positions = utils.dps_to_np([pmc.calc_center(summary_timestamp) for pmc in pmcs.values()]).tolist()
                radii = [pmc.calc_radius(summary_timestamp) for pmc in pmcs.values()]
                weights = [pmc.calc_weight(summary_timestamp) for pmc in pmcs.values()]
                status = np.where(np.array(weights) >= mu, "dense", "normal").tolist()
            else:
                positions = []
                radii = []
                weights = []
                status = []
            id_map = list(pmcs.keys())

            omcs = summary["omcs"]
            if len(omcs.keys()) > 0:
                o_summary_timestamp = list(omcs.values())[0].summary_timestamp
                o_mu = list(omcs.values())[0].mu

                if len(pmcs.keys()) > 0:
                    assert o_mu == mu
                    assert o_summary_timestamp == summary_timestamp

                positions.extend(utils.dps_to_np([omc.calc_center(o_summary_timestamp) for omc in omcs.values()]).tolist())
                radii.extend([omc.calc_radius(o_summary_timestamp) for omc in omcs.values()])
                weights.extend([omc.calc_weight(o_summary_timestamp) for omc in omcs.values()])
                id_map.extend(list(omcs.keys()))
                status.extend(["outlie"] * len(omcs.values()))

            positions = np.array(positions)
            radii = np.array(radii)
            weights = np.array(weights)
            id_map = np.array(id_map)
            status = np.array(status)
        elif online_alg == "streamkmeans_river":
            positions = utils.dps_to_np(summary.values())
            radii = np.zeros(len(summary))
            weights = np.ones(len(summary))
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
        elif online_alg == "birch_sklearn":
            positions = np.array([leaf.centroid_ for leaf in summary.values()])
            sq_radii = [((leaf.squared_sum_ / leaf.n_samples_) - np.dot(leaf.centroid_, leaf.centroid_)) for leaf in summary.values()]
            sq_radii_noneg = np.array([0 if (np.isclose(sq_radii[i], 0) and sq_radii[i] < 0) else sq_radii[i] for i in range(len(sq_radii))])
            radii = sq_radii_noneg**0.5
            weights = np.array([leaf.n_samples_ for leaf in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))

        #print(utils.distance_calc(batch_datapoints, positions, dist="euclidean").shape)
        if len(positions) > 0: # check if actually anything was summarized
            assignment_dists = utils.distance_calc(batch_datapoints, positions, dist="euclidean")
            assignment = np.argmin(assignment_dists, axis=1)
        else:
            assignment = [-1] * len(batch_datapoints)
    elif online_alg == "dstream" or online_alg == "dstream+" or online_alg == "dstream++":
        ex_grid_cell = summary[list(summary.keys())[0]]
        radius = (ex_grid_cell.upper_bound[0] - ex_grid_cell.lower_bound[0]) / 2 + 0.000001 # account for rounding error
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
        #print(upper_bounds)
        #print(lower_bounds)
        #print(positions)
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
        status[np.where(np.array([cell.status for cell in summary.values()]) == "delete")] = "deleted"
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
    elif online_alg == "emcstream" or online_alg == "emcstream_original":
        assignment = np.arange(len(batch_datapoints))  # self assignment
        id_map = assignment
        radii = np.zeros(len(batch_datapoints))
        weights = np.ones(len(batch_datapoints))
        positions = summary["embedding"]
        status = np.array(["dense"] * len(weights))
    elif online_alg == "gbfuzzystream" or online_alg == "gbfuzzystream_original" or online_alg == "stmstream" or online_alg == "stmstream+":
        if online_alg == "gbfuzzystream":
            positions = np.array([fmb.center for fmb in summary.values()])
            radii = np.array([fmb.radius for fmb in summary.values()])
            weights = np.array([len(fmb.samples) for fmb in summary.values()])
            id_map = np.arange(len(summary))
            status = np.array(["dense"] * len(weights))
            sample_ids = [fmb.sample_ids for fmb in summary.values()]
        elif online_alg == "gbfuzzystream_original":
            positions = np.array([ball.get_center() for ball in summary.values()])
            radii = np.array([ball.get_radius() for ball in summary.values()])
            weights = np.array([ball.get_num() for ball in summary.values()])
            id_map = np.array(list(summary.keys()))
            status = np.array(["dense"] * len(weights))
            sample_ids = [ball.data[:,-1] for ball in summary.values()]
        elif online_alg == "stmstream" or online_alg == "stmstream+":
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
        if len(positions) > 0:
            for j in range(len(batch_datapoints)):
                j_id = j + batch_start_id
                found = False
                for h in range(len(sample_ids)):
                    if j_id in sample_ids[h]:
                        assignment.append(h)
                        found = True
                        break
                if not found and len(positions) > 0:  # missing
                    #print(f"{j_id} not found in summary")
                    assignment.append(np.argmin(utils.distance_calc(batch_datapoints[j], positions, dist="euclidean")))
                assert len(assignment) == j + 1
        else:
            assignment = [-1] * len(batch_datapoints)
        assignment = np.array(assignment)
        #print(len(assignment))
    elif online_alg in ["random", "kcentroid", "lwc_square", "protras", "full_batch"]:
        positions = summary["centers"]
        radii = np.zeros(len(positions))
        weights = summary["weights"]
        id_map = np.arange(len(weights))
        status = np.array(["dense"] * len(weights))
        assignment_dists = utils.distance_calc(batch_datapoints, positions, dist="euclidean")
        assignment = np.argmin(assignment_dists, axis=1)
    else:
        raise NotImplementedError(f"{online_alg} is not implemented.")
    min_store = {"id_map": id_map, "center": positions, "radii": radii, "weights": weights, "status": status}
    if do_pseudoclustering:
        if online_alg in ["clustream", "denstream", "hddstream", "dstream", "dstream+", "dstream++", "gbfuzzystream",
                          "stmstream", "stmstream+", "birch", "swclustering", "swclustering+",
                          "clustream_river", "denstream_river", "gbfuzzystream_original", "birch_sklearn"]:  # connected components

            mc_dists = utils.distance_calc(positions, positions, dist="euclidean")
            two_radii = np.array([radii + radii[i] for i in range(len(positions))])
            connected = np.where(mc_dists <= two_radii, True, False)
            partitions = np.arange(len(positions)+1)  # get connected
            for i in range(len(partitions)-1):
                if status[i] != "outlie" and status[i] != "sporad" and status[i] != "sporadic" and status[i] != "deleted" and status[i] != "delete":
                    for j in range(i):
                        if status[j] != "outlie" and status[j] != "sporad" and status[j] != "sporadic" and status[j] != "deleted" and status[j] != "delete":
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
                else:
                    partitions[i] = -1
            partitions[-1] = -1
            return assignment, min_store, partitions[assignment]
        elif online_alg == "emcstream" or online_alg == "emcstream_original":
            pseudoclustering = summary["labels"]
            return assignment, min_store, np.array(pseudoclustering)
        else:  # no pseudo-clustering
            pseudoclustering = assignment
            return assignment, min_store, pseudoclustering
    else:
        return assignment, min_store


import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from clustpy.metrics.clustering_metrics import purity

if __name__ == "__main__":
    stream, labels = load_data("stream_densired")
    #stream = stream[:8500]
    #labels = labels[:8500]
    online_alg = "dstream"
    config_dict = {'heuristic': False, 'epsilon': 0.1, 'B': 1, 'L': 3, 'absorption': 'euclidean', 'batch_budget': 1800, 'batch_size': 1000, 'closeness': 'D3', 'ds': 'densired2', 'max_leaf_entry_num': 100, 'offline': None, 'online': 'birch', 'outlier_batch': 100, 'outlier_ratio': 0.2455760137573}
    config_dict = {'window': 1000, "N": 2000, "epsilon": 0.02, "horizon": 200, "verbose": False, "threshold": 0.05, "branching_factor": 5 }
    config_dict = {'batch_budget': 1800, 'batch_size': 1000, 'ds': 'stream_densired', 'epsilon': 0.4203896618903, 'init_num': 1000, 'k': 100, 'offline': None, 'online': 'stmstream+', 'stride': 500, 'u_radius': 0.0603858461158, 'window': 1000}
    config_dict = {'batch_budget': 1800, 'batch_size': 1000, 'ds': 'har', 'epsilon': 0.9084672533065, 'init_num': 1000, 'k': 100, 'offline': None, 'online': 'stmstream+', 'stride': 100, 'u_radius': 0.0371620941924, 'window': 1000}
    config_dict = {'dim_grid_num': 3}

    stream_alg = get_online_method(online_alg, config_dict, 0)
    batch_size = 1000
    batch_ids = np.arange(0, len(stream), step=batch_size).tolist()
    batch_ids.append(len(stream))
    print(batch_ids)
    timestamps = np.arange(len(stream))
    if online_alg == "emcstream" or online_alg == "emcstream_original" or online_alg == "gbfuzzystream_original" or online_alg == "birch_sklearn":  # needs to run through fully
        stream_alg.learn(stream.tolist(), timestamps.tolist())
        full_summary = stream_alg.get_summary()
    summary = None
    prior_summary = None
    for i in range(len(batch_ids) - 1):
        print(i, len(stream[batch_ids[i]:batch_ids[i + 1]]))
        stream_batch = stream[batch_ids[i]:batch_ids[i + 1]]
        label_batch = labels[batch_ids[i]:batch_ids[i + 1]]
        timestamp_batch = timestamps[batch_ids[i]:batch_ids[i + 1]]
        #print(timestamp_batch)
        if online_alg != "emcstream" and online_alg != "emcstream_original" and online_alg != "gbfuzzystream_original" and online_alg != "birch_sklearn":
            _, event_log = stream_alg.learn(stream_batch.tolist(), timestamp_batch.tolist())
            #print(event_log)
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
        elif online_alg == "gbfuzzystream_original" or online_alg == "birch_sklearn":
            print(batch_ids[i + 1])
            summary = full_summary[batch_ids[i + 1]]
        else:
            summary = {"embedding": full_summary["embedding"][batch_ids[i]:batch_ids[i + 1]],
                       "timestamps": full_summary["timestamps"][batch_ids[i]:batch_ids[i + 1]],
                       "labels": full_summary["labels"][batch_ids[i]:batch_ids[i + 1]]}

        #(summary[list(summary.keys())[-1]])
        #print(summary)
        np.save(f"./storage_test/{online_alg}_store_{i}.npy", summary)
        # https://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
        summary2 = np.load(f"./storage_test/{online_alg}_store_{i}.npy", allow_pickle=True)
        summary2 = summary2.item()
        #print(summary2[list(summary2.keys())[-1]])
        assignment, min_store, pseudoclustering = process_summary(online_alg, summary2, stream_batch, batch_ids[i],
                                                                  True)
        _, pseudoclustering_unq = np.unique(pseudoclustering, return_inverse=True)
        pseudoclustering = np.where(pseudoclustering == -1, -1, pseudoclustering_unq)
        plt.figure(figsize=(30, 30))
        #print(min_store)
        ids = min_store["id_map"]
        print(len(ids))
        centers = min_store["center"]
        weights = min_store["weights"]
        radii = min_store["radii"]
        status = min_store["status"]
        #print(status)

        plt.scatter(x=stream_batch[:, 0], y=stream_batch[:, 1], cmap=plt.get_cmap("jet"), c=pseudoclustering, s=10)
        for j in range(stream_batch.shape[0]):
            plt.text(stream_batch[j][0], stream_batch[j][1], s=str(assignment[j]) + "_" + str(pseudoclustering[j]), c="grey")

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

        assignpurity = purity(labels_true=label_batch, labels_pred=assignment)
        pseudopurity = purity(labels_true=label_batch, labels_pred=pseudoclustering)
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.title(f"assign purity: {assignpurity}; pseudo purity: {pseudopurity}")
        plt.savefig(f"./test_figures_2/{online_alg}_store_{i}.png")
        plt.close()
