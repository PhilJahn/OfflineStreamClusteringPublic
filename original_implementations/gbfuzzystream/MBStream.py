# Original code by xjnine and 920850946 (dmg) , from https://github.com/xjnine/GBFuzzyStream/tree/master

import collections
import copy
import math
import sys
import time
import tracemalloc

import pandas as pd
from sklearn.decomposition import PCA
from original_implementations.gbfuzzystream.HyperballClustering import *
from original_implementations.gbfuzzystream.MicroCluster import MicroBall
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from original_implementations.gbfuzzystream.DPC import get_cluster_DPC
from original_implementations.gbfuzzystream.granular_ball import GranularBall

verbose = False


class MBStream:
    def __init__(self, data, plot_evaluate_flag, lam=1, window=1000, threshold =0.8, m=2, eps=10, verbose=False, batchsize=1000, track_memory=False):
        self.track_memory = track_memory

        self.window = window
        indices = np.reshape(np.arange(len(data)), (-1,1))
        data = np.append(data, indices, axis=1)
        data = np.append(data, indices, axis=1)
        #data = np.append(data,  np.floor(indices/self.window), axis=1)
        #data = np.append(data, np.reshape(np.array([None]*len(data)), (-1,1)), axis=1)
        self.timeSeries = data[:, -1]
        self.timeSeries = np.floor(data[:, -2]/self.window)
        self.timeSpan = 1
        self.data = self.normalized(data)
        self.verbose =verbose
        if self.verbose:
            print(self.data.shape)
        self.begin = self.timeSeries[0]
        self.stop = self.timeSeries[-1]
        self.batchsize = batchsize
        self.timeIndex = 0
        self.lam = lam
        self.m = m #2
        self.eps = eps #10
        self.threshold = threshold  # 0.3
        self.d_ctr = 0
        self.trueLabel = list(map(int, data[:, -2]))
        self.predictions = {}
        self.micro_balls, time, mem = self.init_v1(plot_evaluate_flag)
        self.all_mems = {self.d_ctr: mem[1]}
        self.all_times = {self.d_ctr: time}
        self.all_micro_balls = {self.d_ctr: copy.deepcopy(self.micro_balls)}

    @staticmethod
    def normalized(data):
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 数据缩放
        value = data[:, :-2]
        index = data[:, -1]
        # deactivated for consistency with other methods
        # if len(value[0]) > 2:
        #     #pca = PCA(n_components=2)
        #     #value = pca.fit_transform(value)
        #     value = min_max_scaler.fit_transform(value)
        # else:
        #     value = min_max_scaler.fit_transform(value)
        index = index.reshape(len(index), 1)
        data = np.append(value, index, axis=1)  # 拼接时间戳到数据上（data，index）
        return data

    def get_cluster(self, gb_dict):
        cluster = {}
        cluster_label_list = []
        for i in range(0, len(gb_dict)):
            if gb_dict[i].label not in cluster_label_list:
                cluster_label_list.append(gb_dict[i].label)
        for label in cluster_label_list:
            for key in gb_dict.keys():
                if gb_dict[key].label == label:
                    if label not in cluster.keys():
                        cluster[label] = gb_dict[key].data
                    else:
                        cluster[label] = np.append(cluster[label], gb_dict[key].data, axis=0)
        return cluster

    # v1.0
    def init_v1(self, plot_evaluate_flag):
        init_data = []
        # 收集第一时刻数据放入init_data
        for x in range(0, len(self.data)):
            if self.timeSeries[x] <= self.begin:
                init_data.append(self.data[x])
            else:
                self.timeIndex = x
                break
        if self.verbose:
            print(len(init_data))

        peak_mem = -1
        if self.track_memory:
            tracemalloc.start()
        # https://stackoverflow.com/questions/67702626/how-to-start-stopwatch-in-python


        gb_list, runtime, peak_mem = self.connect_ball_DPC(1, peak_mem, 0,np.array(init_data))
        start = time.time()
        self.d_ctr = len(init_data)
        init_mb_list = []
        for data in gb_list:
            mb = MicroBall(data, None, m=self.m, eps=self.eps)
            mb.init_weight(self.begin, len(data))
            init_mb_list.append(mb)
        if self.verbose:
            print("now have", len(gb_list), "micro-balls")

        # for label, dps in self.get_cluster(gb_dict).items():
        #     for dp in dps:
        #         index = int(dp[-1])
        #         if index in self.predictions.keys():
        #             if self.verbose:
        #                 print("Duplicate Index")
        #         self.predictions[index] = label

        #if plot_evaluate_flag:
        #	gb_plot(gb_dict, [], 1)
        stop = time.time()
        runtime = stop - start
        if self.track_memory:
            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            mem = (-1, -1)

        return init_mb_list, runtime, mem

    @staticmethod
    def get_nearest_micro_ball(sample, micro_balls):
        smallest_distance = sys.float_info.max
        nearest_micro_ball = None
        nearest_micro_ball_index = -1
        for i, micro_ball in enumerate(micro_balls):
            current_distance = np.linalg.norm(micro_ball.center - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_ball = micro_ball
                nearest_micro_ball_index = i
        return nearest_micro_ball_index, nearest_micro_ball, smallest_distance

    def fit_predict(self, plot_evaluate_flag):
        if self.track_memory:
            peak_mem = -1
            tracemalloc.start()
        runtime = 0
        start = time.time()
        t = self.begin + self.timeSpan


        while t <= self.stop:
            new_samples = []
            for i in range(self.timeIndex, len(self.timeSeries)):
                if math.isclose(self.timeSeries[i], t):
                    new_samples.append(self.data[i])
                    self.d_ctr += 1
                    if i == len(self.timeSeries) - 1:
                        break
                    if not math.isclose(self.timeSeries[i + 1], t):
                        self.timeIndex = i + 1
                        break
            if self.verbose:
                print("new_samples", len(new_samples))
            if len(new_samples) == 0:
                if self.verbose:
                    print(f"{t} have no new samples")
            new_samples = np.array(new_samples)
            # Split the data arriving at this moment into spheres
            gb_list_temp = [new_samples]  # Ball set, initially there is only one ball [ [[data1],[data2],...], [[data1],[data2],...],... ]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp, m=self.m, eps=self.eps)  # Particle-sphere division
                ball_number_new = len(gb_list_temp)
                #print("ball_number", ball_number_new, ball_number_old)
                if ball_number_new == ball_number_old:
                    break
            radius = []  # Summarize all sphere radii
            for gb_data in gb_list_temp:
                if len(gb_data) >= 2:  # When there are more than 2 samples in a ball, it is considered a legal ball and its ball radius is collected
                    radius.append(get_radius(gb_data[:, :-1]))
            radius_median = np.median(radius)
            radius_mean = np.mean(radius)
            radius = min(radius_median, radius_mean)
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = minimum_ball(gb_list_temp, radius, m=self.m, eps=self.eps)  # Shrink the ball
                ball_number_new = len(gb_list_temp)
                #print("ball_number2", ball_number_new, ball_number_old)
                if ball_number_new == ball_number_old:
                    break
            for obj in gb_list_temp:
                if len(obj) == 1:
                    stop = time.time()
                    runtime += stop - start
                    if self.track_memory:
                        mem = tracemalloc.get_traced_memory()
                        peak_mem = max(mem[1], peak_mem)
                        tracemalloc.stop()
                    if int(obj[0,-1]) not in self.predictions.keys():
                        self.predictions[int(obj[0,-1])] = -1
                    if self.track_memory:
                        tracemalloc.start()
                    start = time.time()
                    continue
                gb = GranularBall(obj)
                if not self.micro_balls:
                    if self.verbose:
                        print("self.micro_balls is None")
                nearest_micro_ball_index, nearest_micro_ball, smallest_distance = \
                    self.get_nearest_micro_ball(gb.center, self.micro_balls)
                centers = [mb.center for mb in self.micro_balls]
                if nearest_micro_ball is None:
                    if self.verbose:
                        print("nearest_micro_ball is none")
                # ---
                if len(centers) > 1:
                    MIR = min((  # distance between the 1st closest center and the 2nd
                        # 离nearest_micro_ball最近的球 的距离
                        np.linalg.norm(center - nearest_micro_ball.center) for center in centers
                        if center is not nearest_micro_ball.center
                    ))
                else: # default to not merge if MIR unclear
                    MIR = 0
                # if smallest_distance >= radius:
                if nearest_micro_ball is None or (smallest_distance + gb.radius >= MIR and smallest_distance + gb.radius > nearest_micro_ball.radius and smallest_distance + nearest_micro_ball.radius > gb.radius):
                    mb = MicroBall(gb.data, label=None, m=self.m, eps=self.eps)
                    mb.init_weight(t, len(gb.data))
                    self.micro_balls.append(mb)
                else:
                    insert = nearest_micro_ball.insert_ball(gb, t)
                    if not insert:
                        continue
                    else:
                        del self.micro_balls[nearest_micro_ball_index]
                        self.micro_balls.extend(insert)
            temp = []
            for mb in self.micro_balls:
                # 更新权重
                mb.update_weight(t, self.lam)
                if mb.weight > self.threshold:
                    temp.append(mb)
                else:
                    stop = time.time()
                    runtime += stop - start
                    if self.track_memory:
                        mem = tracemalloc.get_traced_memory()
                        peak_mem = max(mem[1], peak_mem)
                        tracemalloc.stop()
                    #print(mb.data)
                    for dp in mb.data:
                        if int(dp[-1]) not in self.predictions.keys():
                            self.predictions[int(dp[-1])] = -1

                    if self.track_memory:
                        tracemalloc.start()
                    start = time.time()
            self.micro_balls = temp

            #print(self.d_ctr)
            if self.d_ctr % self.batchsize == 0 or self.d_ctr == len(self.timeSeries):
                stop = time.time()
                if self.track_memory:
                    mem = tracemalloc.get_traced_memory()
                    peak_mem = max(mem[1], peak_mem)
                    tracemalloc.stop()

                self.all_mems[self.d_ctr] = peak_mem
                self.all_times[self.d_ctr] = runtime + stop - start
                self.all_micro_balls[self.d_ctr] = copy.deepcopy(self.micro_balls)

                if self.track_memory:
                    peak_mem = -1
                    tracemalloc.start()
                runtime = 0
                start = time.time()
            #for mb in self.micro_balls:
            #	print(mb.data)
            #print(self.micro_balls)
            # clusters, gb_list, gb_dict = self.connect_ball_DPC(t)
            # for label, dps in self.get_cluster(gb_dict).items():
            #     for dp in dps:
            #         index = int(dp[-1])
            #         if index not in self.predictions.keys():
            #             self.predictions[index] = str(int(t)) + "_" + str(label)
            #         else:
            #             if self.verbose:
            #                 print("Duplicate Index: ", index)
            # if self.verbose:
            # print(len(self.predictions.keys()))
            t += self.timeSpan
            # if self.verbose:
            #print("now have", len(self.micro_balls), "micro-balls")

    def connect_ball_DPC(self, t, peak_mem, runtime, init_data=None):
        """
        (zt) If init_data is not empty, it is the initialization stage, otherwise it is the intermediate stage.
        Take the center of the sphere as the input data, connect the spheres, and generate clusters.
        :param init_data: Initialization data.
        :return:clusters Initialization data clustering results {cluster labels: [data1, data2,...]}
        """
        start = time.time()
        radius = []  # Summarize all sphere radii
        gb_list_temp = []
        if init_data is not None:
            gb_list_temp = [init_data]  # Ball set, initially there is only one ball [ [[data1],[data2],...], [[data1],[data2],...],... ]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp, m=self.m, eps=self.eps)  # 粒球划分
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            for gb_data in gb_list_temp:
                if len(gb_data) >= 2:  # When there are more than 2 samples in a ball, it is considered a legal ball and its ball radius is collected
                    radius.append(get_radius(gb_data[:, :-1]))
        else:
            for mb in self.micro_balls:
                if len(mb.data) >= 2:
                    radius.append(mb.radius)
                gb_list_temp.append(mb.data)
            if self.verbose:
                print("gb_list_temp", len(gb_list_temp))
                print("micro_balls", len(self.micro_balls))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius = max(radius_median, radius_mean)
        while 1:
            ball_number_old = len(gb_list_temp)
            gb_list_temp = normalized_ball(gb_list_temp, radius, self.m ,self.eps)  # 归一化
            ball_number_new = len(gb_list_temp)
            if ball_number_new == ball_number_old:
                break
        gb_center_list = []
        # noise
        gb_list_temp_no_noise = []
        for gb in gb_list_temp:
            # noise
            if len(gb) > 1:
                gb_center_list.append(gb[:, :-1].mean(0))
                gb_list_temp_no_noise.append(gb)
            else:
                stop = time.time()
                runtime += stop - start
                if self.track_memory:
                    mem = tracemalloc.get_traced_memory()
                    peak_mem = max(mem[1], peak_mem)
                    tracemalloc.stop()
                self.predictions[int(gb[0,-1])] = -1
                if self.track_memory:
                    tracemalloc.start()
                start = time.time()
        #gb_center = np.array(gb_center_list)
        # noise
        gb_list_temp = gb_list_temp_no_noise
        #clusters_label, n = get_cluster_DPC(gb_center)
        # if self.verbose:
        #     print("==================t=" + str(t) + "=================")
        #     print("  c_num: " + str(n))
        # clusters = {}
        # gb_dict = {}
        # print("clusters_label: ", clusters_label, "gb_list_temp: ", gb_list_temp)
        # for i in range(0, len(gb_list_temp)):
        #     gb_dict[i] = GB(gb_list_temp[i], clusters_label[i])
        #     if clusters_label[i] in clusters.keys():
        #         clusters[clusters_label[i]] = np.append(clusters[clusters_label[i]], gb_list_temp[i], axis=0)
        #     else:
        #         clusters[clusters_label[i]] = gb_list_temp[i]
        stop = time.time()
        runtime += stop - start
        return gb_list_temp, peak_mem, runtime


def start(data, plot_evaluate_flag, lam, window):
    M = MBStream(data, plot_evaluate_flag, lam, window)
    M.fit_predict(plot_evaluate_flag)
    od = collections.OrderedDict(sorted(M.predictions.items()))


class OriginalGBFuzzyStream():
    def __init__(self, lam, window, threshold, m, eps, batchsize, track_memory=False):
        self.lam = lam
        self.window = window
        self.batchsize = batchsize
        self.threshold = threshold
        self.m = m
        self.eps = eps
        self.track_memory = track_memory
        self.summaries = None

    def learn(self, data, timestamps):
        plot_evaluate_flag=False
        mb = MBStream(data, plot_evaluate_flag, lam=self.lam, window=self.window, batchsize=self.batchsize, m=self.m, eps=self.eps, track_memory=self.track_memory, verbose=False)
        mb.fit_predict(plot_evaluate_flag)
        self.summaries = mb.all_micro_balls
        return [], [], mb.all_mems, mb.all_times

    def get_summary(self):
        summary_dict = {}
        summary_memory_dict = {}
        for dctr, mbs in self.summaries.items():
            if self.track_memory:
                tracemalloc.start()
            summaries = dict(zip(list(range(len(mbs))), mbs))
            #print("dctr", dctr, len(mbs))
            summary_dict[dctr] = summaries
            if self.track_memory:
                batch_summary_mem = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            else:
                batch_summary_mem = -1
            summary_memory_dict[dctr] = batch_summary_mem
        return summary_dict, summary_memory_dict

