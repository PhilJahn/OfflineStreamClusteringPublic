import math
import time
import tracemalloc
from copy import deepcopy, copy

from matplotlib import pyplot as plt

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer
import numpy as np
from numpy.random import PCG64

class STMStream(StreamClusterer):
    def __init__(self, epsilon=0.1, k=10, window=1000, stride=500, init_num=1000, dist="euclidean", u_radius=0.05, random_state=42, track_memory=False, verbose=False):
        super().__init__()

        self.epsilon = epsilon
        self.k = k
        self.window = window
        self.stride = stride
        self.dist = dist
        self.u_radius = u_radius # unit radius for noise cells

        self.init_num = init_num
        self.initialized = False
        self.init_data = []
        self.init_samples_ids = []
        self.init_timestamps = []

        self.cells = {}
        self.cur_data = []
        self.cur_sample_ids = []
        self.cur_timestamps = []
        self.cell_id = 0

        self.random_state = random_state
        self.random_generator = np.random.Generator(PCG64(random_state))
        self.track_memory = track_memory
        self.verbose = verbose

        self.d_ctr = 0

    def initial_split(self):
        samples = self.init_data
        sample_ids = self.init_samples_ids
        timestamps = self.init_timestamps
        temp_nuclei_ids = self.random_generator.choice(np.arange(len(samples)), size=self.k, replace=False)
        temp_nuclei = samples[temp_nuclei_ids]
        dists = utils.distance_calc(samples, temp_nuclei, self.dist)
        initial_cell_samples = []
        initial_cell_samples_ids = []
        initial_cell_timestamps = []
        for _ in range(self.k):
            initial_cell_samples.append([])
            initial_cell_samples_ids.append([])
            initial_cell_timestamps.append([])
        for i in range(len(samples)):
            ind = np.argmin(dists[i])
            initial_cell_samples[ind].append(samples[i])
            initial_cell_samples_ids[ind].append(sample_ids[i])
            initial_cell_timestamps[ind].append(timestamps[i])
        for j in range(self.k):
            initial_cell_samples[j] = np.array(initial_cell_samples[j])
            initial_cell_samples_ids[j] = np.array(initial_cell_samples_ids[j])
            initial_cell_timestamps[j] = np.array(initial_cell_timestamps[j])
        return initial_cell_samples, initial_cell_samples_ids, initial_cell_timestamps


    def learn_one(self, x, t: int = None):
        if self.track_memory:
            tracemalloc.start()
        # https://stackoverflow.com/questions/67702626/how-to-start-stopwatch-in-python
        start = time.time()

        event_log = ""
        self.d_ctr += 1
        x = np.array(x)
        if not self.initialized:
            if len(self.init_data) < self.init_num:
                self.init_data.append(x)
                self.init_samples_ids.append(self.d_ctr-1)
                self.init_timestamps.append(t)
            if len(self.init_data) == self.init_num:
                self.init_data = np.array(self.init_data)
                self.init_samples_ids = np.array(self.init_samples_ids)
                self.init_timestamps = np.array(self.init_timestamps)
                unfinished = list(range(self.k))
                samples, sample_ids, timestamps = self.initial_split()
                while len(unfinished) > 0:
                    for id in copy(unfinished):
                        if len(samples[id]) == 0:
                            unfinished.remove(id)
                            if self.verbose:
                                print(id, "empty")
                        else:
                            split_index = get_split_index(samples[id], self.dist)
                            cur_samples = samples[id]
                            cur_sample_ids = sample_ids[id]
                            cur_timestamps = timestamps[id]
                            if self.verbose:
                                print(id, split_index, len(samples[id]))
                            if split_index > self.epsilon:
                                id1, id2, samples1, samples2 = perform_split(cur_samples, self.dist)
                                #print(samples1, samples2)
                                samples[id] = cur_samples[samples1]
                                sample_ids[id] = cur_sample_ids[samples1]
                                timestamps[id] = cur_timestamps[samples1]
                                samples.append(cur_samples[samples2])
                                sample_ids.append(cur_sample_ids[samples2])
                                timestamps.append(cur_timestamps[samples2])
                                unfinished.append(len(samples)-1)
                                if self.verbose:
                                    print(id, "split", id1, id2, len(samples1), len(samples2), unfinished)
                            else:
                                unfinished.remove(id)
                                if self.verbose:
                                    print(id, "finished")

                for cell_id in range(len(samples)):
                    cell_samples = samples[cell_id]
                    if len(cell_samples) > 0:
                        cell_samples_ids = sample_ids[cell_id]
                        cell_time = timestamps[cell_id]
                        new_cell = Cell(cell_id, cell_samples, cell_samples_ids, cell_time, self.dist, self.window, self.u_radius)
                        self.cells[cell_id] = new_cell

                for cell_id in self.cells.keys():
                    self.cells[cell_id].update_metrics()
                self.cell_id = len(samples)
                self.initialized = True
                self.init_data = []
                self.init_timestamps = []
                event_log += "initialized;"
        else:
            self.cur_data.append(x)
            self.cur_sample_ids.append(self.d_ctr-1)
            self.cur_timestamps.append(t)
            if (t+1) % self.stride == 0:
                insertion_dict = {}
                for j in range(len(self.cur_data)):
                    dp = self.cur_data[j]
                    dp_id = self.cur_sample_ids[j]
                    closest_in = None
                    closest_dist = np.inf
                    for cell_id in self.cells:
                        cell = self.cells[cell_id]
                        dist = utils.distance_calc(dp,cell.nucleus, dist=self.dist)
                        if closest_dist > dist:
                            closest_dist = dist
                            if dist <= cell.radius:
                                closest_in = cell_id
                            else:
                                closest_in = None
                    if closest_in is not None:
                        if closest_in in insertion_dict.keys():
                            insertion_dict[closest_in].append(j)
                        else:
                            insertion_dict[closest_in] = [j]
                        event_log += f"{self.cur_timestamps[j]} insert {closest_in};"
                    else:
                        new_cell = Cell(self.cell_id, np.array([dp]), np.array([dp_id]), np.array([self.cur_timestamps[j]]), self.dist, self.window, self.u_radius)
                        self.cells[self.cell_id] = new_cell
                        event_log += f"{self.cur_timestamps[j]} new_cell {self.cell_id};"
                        self.cell_id += 1
                for cell_id in insertion_dict.keys():
                    insert_data = np.array(self.cur_data)[insertion_dict[cell_id]].tolist()
                    insert_sample_ids = np.array(self.cur_sample_ids)[insertion_dict[cell_id]].tolist()
                    insert_timestamps = np.array(self.cur_timestamps)[insertion_dict[cell_id]].tolist()
                    self.cells[cell_id].insert(insert_data, insert_sample_ids, insert_timestamps)
                del_ids = []
                for cell_id in self.cells.keys():
                    self.cells[cell_id].age_out(t)
                    if self.cells[cell_id].aged_out or self.cells[cell_id].noised_out:
                        del_ids.append(cell_id)
                        event_log += f"del {cell_id};"
                for del_id in del_ids:
                    self.cells.pop(del_id)
                self.cur_data = []
                self.cur_sample_ids = []
                self.cur_timestamps = []

        stop = time.time()
        if self.track_memory:
            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            mem = (-1, -1)

        return -1, event_log, mem[1], stop - start

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        mems = []
        times = []
        for i in range(len(xs)):
            mc_id, event_log, mem, time = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
            mems.append(mem)
            times.append(time)
        return mc_ids, event_logs, mems, times

    def get_summary(self):
        return self.cells

class Cell:
    def __init__(self, cell_id, samples, sample_ids, timestamps, dist, window, u_radius):
        self.radius = None
        self.nucleus = None
        self.cell_id = cell_id
        self.samples = samples
        self.sample_ids = sample_ids
        self.timestamps = timestamps
        self.dist = dist
        self.window = window
        self.u_radius = u_radius
        self.aged_out = False
        self.noise = False
        self.noised_out = False
        self.update_metrics()
        self.noise = False # reset for new ones
        self.aged_out = False # reset for new ones
        self.noised_out = False # reset for new ones


    def update_metrics(self):
        if len(self.samples) > 1:
            self.nucleus, _, self.radius, _ = get_cell_init_metrics(self.samples, self.dist)
            self.noise = False
        elif len(self.samples) == 1:
            self.nucleus = self.samples[0]
            self.radius = self.u_radius
            if self.noise:
                self.noised_out = True
            else:
                self.noise = True
        else:
            self.aged_out = True
            self.radius = self.u_radius

    def insert(self, samples, sample_ids, timestamps):
        sample_list = self.samples.tolist()
        sample_ids_list = self.sample_ids.tolist()
        timestamp_list = self.timestamps.tolist()
        sample_list.extend(samples)
        sample_ids_list.extend(sample_ids)
        timestamp_list.extend(timestamps)
        if len(samples) > 0:
            self.noise = False
        self.samples = np.array(sample_list)
        self.sample_ids = np.array(sample_ids_list)
        self.timestamps = np.array(timestamp_list)

    def age_out(self, timestamp):
        # samples_old = copy(self.samples)
        relevant_data = np.where(self.timestamps > timestamp - self.window)
        self.samples = self.samples[relevant_data]
        self.sample_ids = self.sample_ids[relevant_data]
        # if len(samples_old) > len(self.samples):
        #     print("aged_out", self.cell_id)
        self.timestamps = self.timestamps[relevant_data]
        self.update_metrics()

    def get_dict(self):
        dic = {}
        dic["cell_id"] = self.cell_id
        dic["nucleus"] = self.nucleus
        dic["radius"] = self.radius
        dic["num_samples"] = len(self.samples)
        dic["sample_ids"] = self.sample_ids
        dic["samples"] = self.samples
        dic["old_timestamp"] = min(self.timestamps)
        dic["new_timestamp"] = max(self.timestamps)
        dic["dist"] = self.dist
        dic["window"] = self.window
        dic["u_radius"] = self.u_radius
        dic["aged_out"] = self.aged_out
        dic["noise"] = self.noise
        return dic

    def __str__(self):
        return str(self.get_dict())

def get_cell_init_metrics(samples, dist):
    dists = utils.distance_calc(samples, samples, dist)
    nucleus_id = np.argmin(np.sum(dists, axis=1))
    nucleus = samples[nucleus_id]
    cell_radius = np.max(dists[nucleus_id])
    return nucleus, nucleus_id, cell_radius, dists


def get_cell_metrics(samples, dist):
    nucleus, nucleus_id, cell_radius, dists = get_cell_init_metrics(samples, dist)
    nucleus_id = np.argmin(np.sum(dists, axis=1))
    cell_range = np.max(dists)
    cell_mean = np.sum(dists[nucleus_id]) / len(samples)
    cell_std = (np.sum(np.square(dists[nucleus_id] - cell_mean))) ** 0.5
    return nucleus, cell_mean, cell_std, cell_range, cell_radius

def get_split_index(samples, dist):
    if len(samples) == 1:
        return 0
    _, cell_mean, cell_std, cell_range, _ = get_cell_metrics(samples, dist)
    si = 1 - math.e ** ((-1*cell_std/cell_mean) * cell_range)
    return si

def perform_split(samples, dist):
    dists = utils.distance_calc(samples, samples, dist)
    argx = np.argmax(dists)
    nucleus1 = argx // dists.shape[0]
    nucleus2 = argx % dists.shape[0]

    #assert dists[nucleus1,nucleus2] == np.max(dists)

    closer_1 = np.where(dists[nucleus1] - dists[nucleus2] <= 0)[0]
    closer_2 = np.where(dists[nucleus1] - dists[nucleus2] > 0)[0]

    return nucleus1, nucleus2, closer_1, closer_2

    best_nucelus_1 = None
    best_nucelus_2 = None
    best_distsum = np.inf
    best_closer_1 = None
    best_closer_2 = None
    for i in range(len(samples)):
        for j in range(len(samples)):
            if i < j:
                closer_i = np.where(dists[i] - dists[j] <= 0)[0]
                closer_j = np.where(dists[i] - dists[j] > 0)[0]
                distsum = np.sum(dists[i][closer_i]) + np.sum(dists[j][closer_j])
                if distsum < best_distsum:
                    best_distsum = distsum
                    best_nucelus_1 = i
                    best_nucelus_2 = j
                    best_closer_1 = closer_i
                    best_closer_2 = closer_j
    return best_nucelus_1, best_nucelus_2, best_closer_1, best_closer_2





import matplotlib.pyplot as plt
import matplotlib.patches as ptc
if __name__ == "__main__":


    make_fig = True
    stream, labels = load_data("stream_densired")  # [[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    stmstream = STMStream()

    #perform_split(stream[:1000], "euclidean")


    for i in range(len(stream)):
        cell_id, event_log = stmstream.learn_one(stream[i], ts[i])
        batch_start = max(0, i + 1 - 1000)
        if (i+1) % 500 == 0 and i > 600:
            #print(i)
            print(i, event_log)
            summary = stmstream.get_summary()
            plt.figure(figsize=(8, 8))
            for id, s in summary.items():
                mc_patch = ptc.Circle((float(s.nucleus[0]), float(s.nucleus[1])), s.radius, alpha=0.4, color="lightgrey")
                plt.gca().add_patch(mc_patch)
                plt.scatter(x=s.nucleus[0], y=s.nucleus[1], alpha=1, c=s.cell_id, s=50)
                plt.text(s.nucleus[0], s.nucleus[1], s=s.cell_id)

                # print(id, s.timestamps)
                # print(id, s.samples)
                #print(s.nucleus in np.array(stream)[batch_start:i+1])
            plt.scatter(x=np.array(stream)[batch_start:i+1][:,0], y=np.array(stream)[batch_start:i+1][:,1], cmap=plt.get_cmap("jet"), c=labels[batch_start:i+1], s=10)
            plt.ylim((0, 1))
            plt.xlim((0, 1))
            plt.savefig(f"test_figures/stmstream/complex9_stmstream_{i}.png", bbox_inches='tight')
            plt.close()
            ctr = 0
            # for j in range(batch_start,i+1):
            #     ctr += 1
            #     cell_id = None
            #     instance = stream[j]
            #     for id, s in summary.items():
            #         if instance in s.samples:
            #             cell_id = id
            #     #print(j, cell_id, i)
            for j in np.arange(batch_start, i+1):
                data_point = stream[j]
                problem = False
                for cell_id, cell in summary.items():
                    if j in cell.sample_ids:
                        if data_point in cell.samples:
                            print(j, "found")
                        else:
                            problem = cell
                    if data_point in cell.samples:
                        if j in cell.sample_ids:
                            print(j, "rev. found")
                        else:
                            problem = cell
                if problem:
                    print(j, data_point)
                    for cell_id, cell in summary.items():
                        if j in cell.sample_ids:
                            print(cell)
                        if data_point in cell.samples:
                            print(cell)





    # pos1 = np.array([1,0])
    # pos2 = np.array([[0,0], [24,24]])
    # dist = utils.distance_calc(pos1, pos2, "cosine")
    # print(dist)