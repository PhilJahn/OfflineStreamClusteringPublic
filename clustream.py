import copy
from timeit import default_timer

import numpy as np

from datahandler import load_data
from online.stream_clustering import StreamClusterer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import scipy.stats as stats


class CluStream (StreamClusterer):
    # max_mc: q
    # init_num: InitNumber
    # mbf: t
    # delta: delta
    # m: m
    def __init__(self, max_mc: int, init_num: int = None, mbf: float = 2.0, m: int = 10, delta: float = 512, random_state: int = 0, b=1, l=0, dist="euclidean", window=None, verbose=False):
        super().__init__()
        self.max_mc = max_mc
        if init_num is not None:
            self.init_num = init_num
        else:
            self.init_num = max_mc
        self.mbf = mbf
        self.delta = delta
        self.m = m

        self.init_storage = []
        self.init_timestamps = []
        self.initialized = False
        self.random_state = random_state
        self.mc_id = 0
        self.mcs = {}

        self.mc_nn = {} # saves the closest mc id for every mc

        self.b = b
        self.l = l
        self.dist = dist

        self.verbose = verbose

        self.t = 0
        self.window = window
        self.d_ctr = 0


    def learn_one(self, x, t: int = None):
        if self.window is not None:
            self.t = self.d_ctr // self.window
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        x = np.array(x)
        event_log = ""
        if not self.initialized:
            self.init_storage.append(x)
            self.init_timestamps.append(self.t)
            if len(self.init_storage) == self.init_num:
                if self.init_num <= self.max_mc: # no need for k-Means if we can just use the instances as initial MCs
                    clustering = list(range(len(self.init_storage)))
                else:
                    kmeans = KMeans(n_clusters=self.max_mc, random_state=self.random_state)
                    clustering = kmeans.fit_predict(self.init_storage)
                for i in range(len(self.init_storage)):
                    if clustering[i] not in self.mcs.keys():
                        self.init_mc(self.init_storage[i], self.init_timestamps[i], nn_dist=-1, mc_id=clustering[i])
                    else:
                        mc = self.mcs[clustering[i]]
                        mc.insert_one(self.init_storage[i], self.init_timestamps[i])
                # build mc_nn[clustering[i]]
                for mc_id, mc in self.mcs.items():
                    assert mc.mc_id == mc_id
                    self.update_mc_nn(mc_id)
                self.initialized = True
                return clustering[-1], "initialized;"
            else:
                return -1, "initialization;"
        else:
            closest_mc_id, nn_dist = self.find_closest_mc_id(x, return_distance=True)
            if nn_dist <= self.mcs[closest_mc_id].boundary: # within the maximum boundary vs outside of maximum boundary, edge case 0 distance with 0 radius (due to duplicates) produces constant influx of micro-clusters, so interpreted as less and equals for insertion
                self.mcs[closest_mc_id].insert_one(x, self.t)
                self.update_mc(self.mcs[closest_mc_id])
                if self.verbose:
                    print("insert", closest_mc_id)
                event_log = "insert " + str(closest_mc_id) + ";"
                return closest_mc_id, event_log
            else:
                # prune relevance timestamp
                del_mc_ids = []
                for mc_id, mc in self.mcs.items():
                    if mc.relevance_stamp < self.t - self.delta:
                        del_mc_ids.append(mc_id)
                for mc_id in del_mc_ids:
                    if self.verbose:
                        print("delete", mc_id, self.mcs[mc_id].mc_history)
                    event_log += "delete " + str(mc_id) + ";"
                    self.delete_mc(mc_id)

                # merge closest
                if len(self.mcs) >= self.max_mc:
                    nmc1 = None
                    nmc2 = None
                    min_dist = np.inf
                    for mc_id, mc in self.mcs.items():
                        if mc_id in self.mc_nn.keys():
                            mc2_id = self.mc_nn[mc_id]
                            dist = mc.nn_dist
                            # for sanity check
                            dist2 = mc.get_distance(self.mcs[mc2_id].mean)

                            ##TODO: drop assert for final version
                            #mc3_id, dist3 = self.find_closest_mc_id(mc.mean, [mc_id], return_distance=True)
                            #print(mc_id, t, ":", mc2_id, dist, dist2, mc3_id, dist3)
                            #print(self.mcs[mc_id], self.mcs[mc2_id], self.mcs[mc3_id])
                            #assert mc2_id == mc3_id
                            assert dist == dist2
                            #assert dist == dist3
                        else:
                            mc2_id, dist = self.find_closest_mc_id(mc.mean, [mc_id], return_distance=True)
                        if dist < min_dist:
                            if self.mcs[mc_id].n > self.mcs[mc2_id].n:
                                nmc1 = self.mcs[mc_id]
                                nmc2 = self.mcs[mc2_id]
                            else:
                                nmc2 = self.mcs[mc_id]
                                nmc1 = self.mcs[mc2_id]
                            min_dist = dist
                    # merge into bigger one (always nmc_1)
                    nmc1.merge(nmc2)
                    self.update_mc(nmc1)
                    self.delete_mc(nmc2.mc_id)
                    if self.verbose:
                        print("merge", nmc1.mc_id, nmc2.mc_id, flush=True)
                    event_log += "merge " + str(nmc1.mc_id) + ";"
                    del nmc2
                assert len(self.mcs) < self.max_mc
                new_mc_id = self.init_mc(x, self.t, nn_dist)
                new_mc = self.mcs[new_mc_id]
                self.update_mc(new_mc)
                event_log += "new " + str(new_mc_id) + ";"
                return new_mc_id, event_log

    def update_mc(self, updated_mc):
        updated_mc_id = updated_mc.mc_id
        self.update_mc_nn(updated_mc_id)
        #print(self.mc_nn)
        for mc_id, closest_mc_id in self.mc_nn.items():
            if mc_id != updated_mc_id:
                mc = self.mcs[mc_id]
                dist = mc.get_distance(updated_mc.mean)
                if closest_mc_id == updated_mc_id:
                    #print(mc_id, "closest to ", updated_mc_id)
                    if dist > mc.nn_dist: #farther apart, need to find new closest
                        self.update_mc_nn(mc_id)
                    else:
                        mc.update_nn_dist(dist)
                else:
                    if dist < mc.nn_dist or (dist == mc.nn_dist and updated_mc_id < self.mc_nn[mc_id]): #closer than prior closest
                        self.mc_nn[mc_id] = updated_mc_id
                        mc.update_nn_dist(dist)


    def delete_mc(self, del_id):
        self.mcs.pop(del_id)
        if del_id in self.mc_nn.keys():
            self.mc_nn.pop(del_id)
        if del_id in self.mc_nn.values():
            for mc_id, closest_mc_id in self.mc_nn.items():
                if closest_mc_id == del_id:
                    self.update_mc_nn(mc_id)

    def update_mc_nn(self, mc_id):
        mc = self.mcs[mc_id]
        nn, dist = self.find_closest_mc_id(mc.mean, [mc_id], return_distance=True)
        self.mc_nn[mc_id] = nn
        mc.update_nn_dist(dist)

    def find_closest_mc_id(self, pos, exclusion=None, return_distance=False):
        min_dist = np.inf
        closest_mc_id = None
        if exclusion is None:
            exclusion = []
        for mc_id, mc in self.mcs.items():
            if mc_id not in exclusion:
                dist = mc.get_distance(pos)
                if dist < min_dist:
                    closest_mc_id = mc_id
                    min_dist = dist
        if return_distance:
            return closest_mc_id, min_dist
        else:
            return closest_mc_id


    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs


    def init_mc(self, instance, timestamp, nn_dist, mc_id=None):
        if mc_id is None:
            mc_id = self.mc_id
        assert mc_id not in self.mcs.keys()
        cf2x = []
        cf1x = []
        cf2t = timestamp**2
        cf1t = timestamp
        for d in range(len(instance)):
            cf2x.append(instance[d] ** 2)
            cf1x.append(instance[d])
        # undecayed since just initialized, even if decay
        mc = CluStreamMicroCluster(mc_id=mc_id, cf2x=cf2x, cf1x=cf1x, cf2t=cf2t, cf1t=cf1t, n=1, t=timestamp, count=1,
                                   m = self.m, nn_dist = nn_dist,  mbf=self.mbf, b=self.b, l=self.l, dist=self.dist)
        self.mcs[mc_id] = mc
        self.mc_id = max(self.mcs.keys()) + 1
        return mc_id

    def get_summary(self):
        return copy.deepcopy(self.mcs)


class CluStreamMicroCluster:
    def __init__(self, mc_id, cf2x, cf1x, cf2t, cf1t, n, t, count, nn_dist, m, mbf, b, l, dist):

        """
        :param mc_id: micro-clustering id -> unique
        :param cf2x: CF2X: Sum of Squares for instances
        :param cf1x: CF2X: Linear Sum for instances
        :param cf2t: CF2T: Sum of Squares for timesteps
        :param cf1t: CF1T: Linear Sum for timesteps
        :param n: number of instances
        :param t: current timestamp
        :param count: historical numer of instances
        :param mbf: maximum boundary factor (hyperparameter of main method)
        :param b: b for exponential decay (hyperparameter of main method)
        :param l: lambda for exponential decay (hyperparameter of main method)
        :param dist: distance function (hyperparameter of main method)
        """


        #content
        self.mc_id = mc_id
        self.cf2x = copy.deepcopy(cf2x)
        self.cf1x = copy.deepcopy(cf1x)
        self.cf2t = cf2t
        self.cf1t = cf1t
        self.n = n
        self.last_update = t

        #helpers
        self.count = count
        self.mc_history = [mc_id]
        self.nn_dist = nn_dist

        #hyperparameters
        self.m = m
        self.b = b
        self.l = l
        self.mbf = mbf
        self.dist = dist


        #calculated metrics
        self.mean = None
        self.stdev = None
        self.boundary = None
        self.metric_update = None
        self.relevance_stamp = None
        self.recalc_metrics(self.last_update) # calculate mean + stdev


    def insert(self, n2, count2, cf2x2, cf1x2, cf2t2, cf1t2, timestamp, mc_history2):
        max_ts = max(timestamp,self.last_update)
        dcf_1 = self.b ** (-self.l * (max_ts - self.last_update))
        dcf_2 = self.b ** (-self.l * (max_ts - timestamp))

        for d in range(len(cf2x2)):
            self.cf2x[d] = dcf_1 * self.cf2x[d] + dcf_2 * cf2x2[d]
            self.cf1x[d] = dcf_1 * self.cf1x[d] + dcf_2 * cf1x2[d]
        self.n = dcf_1 * self.n + dcf_2 * n2

        # no aging on timestamps
        self.cf2t += cf2t2
        self.cf1t += cf1t2
        self.count += count2

        self.last_update = max_ts
        self.recalc_metrics(self.last_update)

        self.mc_history.extend(mc_history2) # if you have duplicates, there is something wrong

    def insert_one(self, instance, timestamp):
        n2 = 1
        mc_history2 = []
        cf2x2 = []
        cf1x2 = []
        cf2t2 = timestamp**2
        cf1t2 = timestamp
        for d in range(len(instance)):
            cf2x2.append(instance[d] ** 2)
            cf1x2.append(instance[d])
        self.insert(n2,n2,cf2x2,cf1x2,cf2t2,cf1t2,timestamp, mc_history2)

    def merge(self, mc2):
        n2 = mc2.n
        count2 = mc2.count
        mc_history2 = mc2.mc_history
        cf2x2 = copy.deepcopy(mc2.cf2x)
        cf1x2 = copy.deepcopy(mc2.cf1x)
        cf2t2 = mc2.cf2t
        cf1t2 = mc2.cf1t
        timestamp = mc2.last_update
        self.insert(n2,count2,cf2x2,cf1x2,cf2t2,cf1t2,timestamp, mc_history2)

    def subtract(self, mc2):
        n2 = -1 * mc2.n
        count2 = -1 * mc2.count
        mc_history2 = mc2.mc_history
        cf2x2 = -1 * np.array(mc2.cf2x)
        cf1x2 = -1 * np.array(mc2.cf1x)
        cf2t2 = -1 * mc2.cf2t
        cf1t2 = -1 * mc2.cf1t
        #print(cf2x2, cf1x2, cf2t2, cf1t2)
        timestamp = mc2.last_update
        self.insert(n2,count2,cf2x2,cf1x2,cf2t2,cf1t2,timestamp, mc_history2)

    def get_dict(self):
        dic = {}
        dic['mc_id'] = self.mc_id
        dic['mc_history'] = self.mc_history
        dic['cf2x'] = self.cf2x
        dic['cf1x'] = self.cf1x
        dic['cf2t'] = self.cf2t
        dic['cf1t'] = self.cf1t
        dic['n'] = self.n
        dic['last_update'] = self.last_update
        dic['count'] = self.count
        dic['mean'] = self.mean
        dic['stdev'] = self.stdev
        dic['boundary'] = self.boundary
        dic['relevance_stamp'] = self.relevance_stamp
        dic['nn_dist'] = self.nn_dist
        return dic

    def get_distance(self, pos):
        if self.dist == "euclidean":
            return np.sqrt(np.sum(np.square(self.mean - pos)))
        elif self.dist == "manhattan":
            return np.sum(np.abs(self.mean - pos))
        else:
            return pairwise_distances(self.mean.reshape(1, -1), pos.reshape(1, -1), self.dist)[0][0]

    def update_nn_dist(self, nn_dist):
        self.nn_dist = nn_dist
        if self.count == 1:
            self.boundary = nn_dist

    def recalc_metrics(self, timestamp):
        self.mean = np.array(self.cf1x) / self.n
        self.stdev = ((np.array(self.cf2x)/self.n) - (np.array(self.cf1x)/self.n)**2)**0.5
        if self.count < 1:
            self.relevance_stamp = 0
        elif self.count < 2*self.m:
            self.relevance_stamp = self.cf1t/self.count
        else:
            time_mean = self.cf1t / self.count
            time_std = ((self.cf2t/self.count) - (self.cf1t/self.count)**2)**0.5
            ppf = 1-(self.m/(2.0*self.count))
            #https://stackoverflow.com/questions/65988123/how-to-compute-the-percentiles-from-a-normal-distribution-in-python
            dist = stats.norm(loc=time_mean, scale=time_std)
            self.relevance_stamp = dist.ppf(ppf)
        if self.count == 0:
            self.boundary = 0 # no micro-cluster anymore
        elif self.count > 1: # only recalculate if more than one instance in micro cluster, otherwise closest micro-cluster
            self.boundary = np.max(self.mbf * self.stdev)
        else:
            self.boundary = self.nn_dist
        self.metric_update = timestamp
        assert self.metric_update == self.last_update

    def __str__(self):
        return str(self.get_dict())

from online.clustream_river import CluStreamRiver
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

if __name__ == '__main__':

    stream, labels = load_data("kddcup")#[[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))

    #{'batch_budget': 1800, 'batch_size': 6000, 'delta': 1713,
    # 'ds': 'kddcup', 'init_num': 2000, 'm': 11, 'mbf': 3.4549739469849, 'mmc': 100, 'offline': None, 'online': 'clustream'}


    # clustream_river = CluStreamRiver(max_micro_clusters=100)
    # start = default_timer()
    # for i in range(len(stream)):
    #     dp = dict(enumerate(stream[i]))
    #     clustream_river.learn_one(dp)
    #     if i % 200 == 0:
    #         print(i)
    # final_time_river = default_timer() - start
    make_fig = False
    clustream = CluStream(max_mc=100, init_num=2000, mbf=3.4549739469849, delta=1713, m=11, verbose=True)
    start = default_timer()
    for i in range(len(stream)):
        nmc_id, event_log = clustream.learn_one(stream[i], ts[i])
        #print(i, event_log)
        mcs = clustream.get_summary()
        if i > 100 and make_fig:
            merge_id = None
            if "merge" in event_log:
                event_log_split = event_log.split(";")
                for event in event_log_split:
                    if "merge" in event:
                        merge_id = event.split(" ")[1]

            plt.figure(figsize=(8,8))
            batch_start = max(0, i + 1 - 500)
            for mc_id, mc in mcs.items():
                #print(mc.mean)
                mc_patch = ptc.Circle((float(mc.mean[0]), float(mc.mean[1])), mc.boundary, alpha=0.4, color="lightgrey")
                plt.scatter(x=mc.mean[0], y=mc.mean[1], c=mc_id, alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                if merge_id is not None:
                    if int(merge_id) == int(mc_id):
                        print("merged")
                        plt.scatter(x=mc.mean[0], y=mc.mean[1], c="pink", alpha=1, marker="x", s=150)
            plt.scatter(x=np.array(stream)[batch_start:i+1][:,0], y=np.array(stream)[batch_start:i+1][:,1], cmap=plt.get_cmap("jet"), c=labels[batch_start:i+1], s=10)
            plt.scatter(x=np.array(stream)[i:i+1][:,0], y=np.array(stream)[i:i+1][:,1], c="pink", s=25)
            plt.savefig(f"test_figures/clustream/complex9_clustream_{i}.png", bbox_inches='tight')
            plt.close()
        #for mc_id, mc in mcs.items():
        #    print(mc)
        #if i % 200 == 0:
        #    print(i)
        #    mcs = clustream.get_summary()
        #    for mc_id, mc in mcs.items():
        #        print(mc)
    final_time_own = default_timer() - start

    #print(final_time_river, final_time_own, final_time_river-final_time_own)


    # N = 5
    # LS = [28]
    # SS = [208]
    # lst = 55
    # sst = 979
    # t = 30
    # alpha = 1.0
    # m=3
    #
    # mc = CluStreamMicroCluster(mc_id=0, cf1t=lst, cf2t=sst, cf1x=LS, cf2x=SS, n=N, t=t, mbf=alpha, count=N, m=m, b=2, l=0.5, dist="euclidean")
    # print(mc)
    # mc.insert_one([8], 30)
    #
    # mc.insert_one([11], 36)
    # print(mc)
    #
    # mc2 = CluStreamMicroCluster(mc_id=0, cf1t=lst, cf2t=sst, cf1x=LS, cf2x=SS, n=N, t=t, mbf=alpha, count=N, m=m, b=2, l=0.5, dist="euclidean")
    # print(mc2)
    # mc4 = copy.deepcopy(mc)
    # mc4.subtract(mc2)
    # print(mc4)
    # print(mc)

    # mc3 = CluStreamMicroCluster(mc_id=0, cf1t=30, cf2t=900, cf1x=[8], cf2x=[64], n=1, t=30, mbf=alpha, count=1, m=m, b=2, l=0.5, dist="euclidean")
    # mc3.insert_one([11], 36)
    # print(mc3)
