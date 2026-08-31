import argparse
import copy
import os
import time
import tracemalloc

import numpy as np
from numpy.random import PCG64
from sklearn.cluster import KMeans, Birch
from sklearn.metrics.pairwise import distance_metrics
from sklearn.neighbors import NearestNeighbors

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer


class NotEnoughDataError(ValueError):
    pass

def random_subsampling(data, count, seed=0):
    generator = np.random.Generator(PCG64(seed))
    rand_subset_indices = np.sort(generator.choice(len(data), size=count, replace=False))
    X_rand_subset = data[rand_subset_indices]
    return X_rand_subset


# Saves centroid for k-Means clusters
def kcentroid_subsampling(data, count, seed=0):
    kmeans = KMeans(n_clusters=count, random_state=seed)
    clustering = kmeans.fit_predict(data)
    X_kcentroid_subset = np.array(kmeans.cluster_centers_)
    return X_kcentroid_subset


# reimplemented from Scalable k-Means Clustering via Lightweight Coresets
def lwc_square_subsampling(data, count, seed=0, distance_function="euclidean"):
    mu = np.mean(data, axis=0).reshape(1, -1)
    distance_metric = distance_metrics()[distance_function]
    dists = distance_metric(mu, data)[0]
    #print(dists)
    probs = []
    divisor = np.sum(dists ** 2)
    #print(divisor)
    for i in range(len(data)):
        prob = 0.5 / len(data)
        if divisor > 0:
            prob += 0.5 * (dists[i] ** 2) / divisor
        else:
            prob *= 2
        #print(prob)
        probs.append(prob)
    generator = np.random.Generator(PCG64(seed))
    lwc_subset_indices = np.sort(generator.choice(len(data), size=count, replace=False, p=probs))
    X_lwc_subset = data[lwc_subset_indices]
    return X_lwc_subset


# better reimplementation from ProTraS: A probabilistic traversing sampling algorithm, altered to use ratio rather than epsilon for consistency with other methods
def protras_subsampling(data, count, distance_function="euclidean"):
    distance_metric = distance_metrics()[distance_function]
    #data = T, pattern = T(y), t_n_s = T \ S
    pattern = {}
    t_n_s = list(range(len(data)))

    virtual_start = np.min(data, axis=0).reshape(1, -1)
    #print(virtual_start)
    closest_i = -1
    closest_distance = np.inf

    for i in range(len(data)):
        dist = distance_metric(virtual_start, data[i].reshape(1, -1))[0][0]
        if dist < closest_distance:
            closest_distance = dist
            closest_i = i
    s = [closest_i]
    pattern[closest_i] = []
    rev_pattern = {}
    t_n_s.remove(closest_i)

    cur_k = closest_i
    for l in range(len(data)):
        rev_pattern[l] = cur_k
        pattern[cur_k].append(l)
    dist_matrix = {}
    dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[
        0]  # no recalculation of distances in every loop
    #print(dist_matrix[cur_k])
    #print(pattern, rev_pattern)

    while len(s) < count:
        for l in t_n_s:  # only update if newest it closer -> no n^2 loop here
            if dist_matrix[cur_k][l] <= dist_matrix[rev_pattern[l]][l]:  # no impact over <, but addresses first loop
                pattern[rev_pattern[l]].remove(l)
                pattern[cur_k].append(l)
                rev_pattern[l] = cur_k
        maxwd = 0
        xmax = -1
        ymax = -1
        for k in s:
            max_dist_k = 0
            max_m = -1
            for m in list(set(pattern[k]).intersection(set(t_n_s))):
                dist_m = dist_matrix[k][m]
                if dist_m > max_dist_k:
                    max_dist_k = dist_m
                    max_m = m
                pk = len(pattern[k]) * max_dist_k
                if maxwd < pk:
                    maxwd = pk
                    ymax = k
                    xmax = max_m
        cur_k = xmax
        if cur_k != -1:
            s.append(cur_k)
            #print(len(s), s, pattern)
            #print(cur_k)
            pattern[rev_pattern[cur_k]].remove(cur_k)
            pattern[cur_k] = [cur_k]
            rev_pattern[cur_k] = cur_k
            t_n_s.remove(cur_k)
            dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0]
        else:
            break

    X_protras_subset = data[s]
    return X_protras_subset


class BatchSummarizer(StreamClusterer):
    def __init__(self, sampler, count, random_state=0, distance_function="euclidean", track_memory=False):
        super().__init__()
        self.sampler = sampler
        self.seed = random_state
        self.count = count
        self.distance_function = distance_function
        self.summary = {}
        self.track_memory = track_memory

    def learn_one(self, x, t: int) -> int:
        raise Exception("No incremental learning")

    def learn(self, xs: list, ts) -> list:
        mems = []
        times = []
        if self.track_memory:
            tracemalloc.start()
        start = time.time()
        data_np = np.array(xs)
        sample_count = min(self.count, len(xs))
        if self.sampler == "batch_random":
            centers = random_subsampling(data_np, sample_count, self.seed)
        elif self.sampler == "batch_kcentroid":
            centers = kcentroid_subsampling(data_np, sample_count, self.seed)
        elif self.sampler == "batch_lwc_square":
            centers = lwc_square_subsampling(data_np, sample_count, self.seed, self.distance_function)
        elif self.sampler == "batch_protras":
            centers = protras_subsampling(data_np, sample_count, self.distance_function)
        elif self.sampler == "full_batch":
            self.summary = {"centers": data_np, "weights": np.ones(len(xs))}
            if self.track_memory:
                mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                mem = (-1, -1)
            stop = time.time()
            times.append(stop - start)
            return [-1, "", mem, times]
        else:
            raise NotImplementedError(self.sampler)

        # nns = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers).kneighbors(data_np, return_distance=False)
        # nn_ids, weights = np.unique(nns, return_counts=True)
        # self.summary = {"centers": np.array(centers[nn_ids]), "weights": np.array(weights)}
        weights = np.zeros(len(centers))
        for dp_np in data_np:
            dists = utils.distance_calc(dp_np, centers, self.distance_function)
            nn = np.argmin(dists)
            weights[nn] += 1

        non_zero = np.where(weights > 0)[0]

        self.summary = {"centers": np.array(centers[non_zero]), "weights": weights[non_zero]}
        if self.track_memory:
            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            mem = (-1, -1)
        stop = time.time()
        mems.append(mem)
        times.append(stop - start)
        #assert len(weights) == sample_count
        return [-1, "", mems, times]

    def get_summary(self):
        return self.summary


class WindowSummarizer(StreamClusterer):
    def __init__(self, sampler, count, window_size, stride, selection_chance, random_state=0,
                 distance_function="euclidean", track_memory=False):
        super().__init__()
        self.sampler = sampler
        self.seed = random_state
        self.count = count
        self.distance_function = distance_function
        self.window_size = window_size
        self.selection_chance = selection_chance
        self.stride = stride
        self.selection_generator = np.random.Generator(PCG64(random_state * 100))
        self.summary = {}
        self.data = []
        self.d_ctr = -1
        self.track_memory = track_memory

    def learn_one(self, x, t: int):
        if self.track_memory:
            tracemalloc.start()
        # https://stackoverflow.com/questions/67702626/how-to-start-stopwatch-in-python
        start = time.time()
        self.d_ctr += 1
        event_log = ""
        rnd = self.selection_generator.random()
        if rnd < self.selection_chance:
            event_log += f"{self.d_ctr} stored;"
            self.data.append(x)
            #print(len(self.data))
            if len(self.data) > self.window_size:
                self.data = self.data[1:]
        if (t + 1) % self.stride == 0:
            if len(self.data) == 0:
                raise NotEnoughDataError
            sample_count = min(self.count, len(self.data))
            data_np = np.array(self.data)
            if self.sampler == "window_random":
                centers = random_subsampling(data_np, sample_count, self.seed)
            elif self.sampler == "window_kcentroid":
                centers = kcentroid_subsampling(data_np, sample_count, self.seed)
            elif self.sampler == "window_lwc_square":
                centers = lwc_square_subsampling(data_np, sample_count, self.seed, self.distance_function)
            elif self.sampler == "window_protras":
                centers = protras_subsampling(data_np, sample_count, self.distance_function)
            else:
                raise NotImplementedError(self.sampler)

            #dists = utils.distance_calc(data_np, centers, self.distance_function)
            # nns = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers).kneighbors(data_np, return_distance=False)
            # #nn = np.argmin(dists, axis=1)
            # nn_ids, weights = np.unique(nns, return_counts=True)
            weights = np.zeros(len(centers))
            for dp_np in data_np:
                dists = utils.distance_calc(dp_np, centers, self.distance_function)
                nn = np.argmin(dists)
                weights[nn] += 1

            non_zero = np.where(weights > 0)[0]

            self.summary = {"centers": np.array(centers[non_zero]), "weights": weights[non_zero]}

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
        return self.summary


import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_points, labels = load_data("stream_densired")
    #data, labels = data[:1000], labels[:1000]
    timestamps = np.arange(len(data_points)).tolist()
    batch_ids = np.arange(0, len(data_points), step=1000).tolist()

    summarizer = WindowSummarizer(sampler="lwc_square", count=100, window_size=1000, stride=1000, selection_chance=0.5)
    for batch in range(len(batch_ids) - 1):
        stream_batch = data_points[batch_ids[batch]:batch_ids[batch + 1]]
        stream_eval_batch = data_points[batch_ids[max(0, batch - 1)]:batch_ids[batch + 1]]
        label_eval_batch = labels[batch_ids[max(0, batch - 1)]:batch_ids[batch + 1]]
        timestamp_batch = timestamps[batch_ids[batch]:batch_ids[batch + 1]]
        summarizer.learn(stream_batch, timestamp_batch)
        summary = summarizer.get_summary()

        centers = summary["centers"]
        weights = summary["weights"]

        plt.figure(figsize=(10, 10))
        plt.scatter(stream_eval_batch[:, 0], stream_eval_batch[:, 1], alpha=0.5, c="blue")
        plt.scatter(centers[:, 0], centers[:, 1], s=weights * 5, c="red")
        plt.show()
