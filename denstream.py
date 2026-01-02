import math
from copy import copy, deepcopy

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer


class DenStream(StreamClusterer):

    # init_num: initN
    def __init__(self, init_num: int = 1000, mu: int = 10, beta: float = 0.2, epsilon: float = 16.0, b=2,
                 l: float = 0.25,
                 stream_speed: int = 1000, alpha=None, heuristic=False, verbose=False):
        super().__init__()
        self.t = 0
        self.d_ctr = 0
        self.verbose = verbose
        self.mc_id = 0
        self.stream_speed = stream_speed
        self.init_num = init_num
        self.init_storage = []
        self.init_timestamps = []
        self.initialized = False

        if not (1 >= beta > 0):
            raise ValueError("beta must be between 0 and 1")
        if not (mu * beta > 1):
            raise ValueError("mu * beta must be more than 1")

        self.heuristic = heuristic
        self.epsilon = epsilon
        self.mu = mu
        if self.heuristic:
            if not (0 < alpha < 1):
                raise ValueError("alpha must be in range ]0,1[")
            # later set epsilon to alpha * min DBSCAN cluster distance and mu to average number in epsilon range,
            # but need initial values for DBSCAN

        self.beta = beta
        self.l = l
        self.b = b

        self.pmcs = {}
        self.omcs = {}

        self.fading_time = math.ceil(1 / l * math.log((beta * mu / (beta * mu - 1)), b))

        print(self.fading_time)

    def initialization(self):
        if self.heuristic:
            dbscan = DBSCAN(eps=self.epsilon, min_samples=self.mu)
            clustering = dbscan.fit_predict(self.init_storage)
            non_noise = np.where(clustering != -1)[0]
            #print(clustering)
            #print(non_noise)
            if len(np.unique(clustering[non_noise])) > 1:  # heuristic doesn't work for only one cluster
                closest_dist = np.inf
                for i in non_noise:
                    cluster_id = clustering[i]
                    other_cluster_ids = np.where(clustering != cluster_id)[0]
                    for j in other_cluster_ids:
                        if j in non_noise:
                            dist = utils.distance_calc(self.init_storage[i], self.init_storage[j],
                                                       dist=self.distance)
                            if dist < closest_dist:
                                closest_dist = dist
                self.epsilon = closest_dist
        # DBSCAN not needed for actual intialization, just range NNs
        _, eps_nns = NearestNeighbors(radius=self.epsilon).fit(self.init_storage).radius_neighbors(
            self.init_storage, radius=self.epsilon)
        # print(eps_nns)
        indices = list(range(len(self.init_storage)))
        sizes = {}
        eps_nns_list = []
        for i in range(len(eps_nns)):
            sizes[i] = len(eps_nns[i])
            eps_nns_list.append(sorted(eps_nns[i].tolist()))
        if self.heuristic:
            cluster_eps_nn_sizes = np.array(list(sizes.values()))[non_noise]
            self.mu = np.floor(np.mean(cluster_eps_nn_sizes))
            if self.mu * self.beta < 1:
                self.beta = 1.0001 / self.mu

        #print(sizes)
        max_nn = max(sizes.values())
        #print(self.beta * self.mu)
        while (max_nn > self.beta * self.mu):
            mc_instances = []
            mc_timestamps = []
            for i in range(len(eps_nns)):
                if len(eps_nns_list[i]) == max_nn:
                    cur_nns = copy(eps_nns_list[i])
                    # print(cur_nns)
                    # print(indices)
                    for nn in cur_nns:
                        mc_instances.append(self.init_storage[nn])
                        mc_timestamps.append(self.init_timestamps[nn])
                        for nn_nn in copy(eps_nns_list[nn]):
                            # print("remove", nn, "in", nn_nn)
                            # print(eps_nns_list[nn_nn])
                            eps_nns_list[nn_nn].remove(nn)
                            sizes[nn_nn] -= 1
                            if nn_nn != nn:
                                eps_nns_list[nn].remove(nn_nn)
                                sizes[nn] -= 1
                        indices.remove(nn)
                    new_mc = self.make_mc(mc_instances[0], mc_timestamps[0])
                    for i in range(1, len(mc_instances)):
                        instance = mc_instances[i]
                        cf1, cf2, w = instance_to_denstream_features(instance)
                        new_mc.insert(cf1, cf2, w, mc_timestamps[i])
                    if new_mc.status == "outlier":
                        self.omcs[new_mc.mc_id] = new_mc
                    else:
                        self.pmcs[new_mc.mc_id] = new_mc
                    break
            max_nn = max(sizes.values())
        self.initialized = True

    def learn_one(self, x, t: int = None):
        if self.stream_speed is not None:
            self.t = self.d_ctr // self.stream_speed
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        event_log = ""
        x = np.array(x)

        if not self.initialized:
            if len(self.init_storage) < self.init_num:
                self.init_storage.append(x)
                self.init_timestamps.append(self.t)
            if len(self.init_storage) == self.init_num:
                self.initialization()
                return -1, "initialized;"
            else:
                return -1, "initialization;"
        else:
            nn_pmc = self.find_closest_mc(x, True)
            if nn_pmc is None or not nn_pmc.potentially_insert(x, self.t):
                nn_omc = self.find_closest_mc(x, False)
                if nn_omc is None or not nn_omc.potentially_insert(x, self.t):
                    new_omc = self.make_mc(x, self.t)
                    new_omc_id = new_omc.mc_id
                    self.omcs[new_omc_id] = new_omc
                    mc_id = new_omc_id
                    event_log += "new O" + str(new_omc_id) + ";"
                else:
                    nn_omc_id = nn_omc.mc_id
                    mc_id = nn_omc_id
                    event_log += "insert O" + str(nn_omc_id) + ";"
                    nn_omc.calc_status(self.t)
                    if nn_omc.status != "outlier":
                        self.pmcs[nn_omc_id] = self.omcs.pop(nn_omc_id)
                        event_log += "upgrade O" + str(nn_omc_id) + ";"
            else:
                mc_id = nn_pmc.mc_id
                event_log += "insert P" + str(mc_id) + ";"
        if self.t % self.fading_time == 0:
            if self.stream_speed is None or self.d_ctr % self.stream_speed == 0:
                #print("check time")
                del_pmc_ids = []
                for pmc_id, pmc in self.pmcs.items():
                    pmc.calc_status(self.t)
                    #print(pmc)
                    if pmc.status == "outlier":
                        del_pmc_ids.append(pmc_id)
                        event_log += "delete P" + str(pmc_id) + ";"
                for del_pmc_id in del_pmc_ids:
                    self.pmcs.pop(del_pmc_id)
                del_omc_ids = []
                for omc_id, omc in self.omcs.items():
                    omc.calc_status(self.t)
                    #print(omc)
                    if omc.status == "outlier":
                        if omc.xi > omc.w:
                            del_omc_ids.append(omc_id)
                            event_log += "delete O" + str(omc_id) + ";"
                    else:
                        del_omc_ids.append(omc_id)
                        self.pmcs[omc_id] = omc
                        event_log += "upgrade O" + str(omc_id) + ";"
                for del_omc_id in del_omc_ids:
                    self.omcs.pop(del_omc_id)
        return mc_id, event_log

    def find_closest_mc(self, pos, pmc=True):
        min_dist = np.inf
        closest_mc = None
        if pmc:
            mcs = self.pmcs.items()
        else:
            mcs = self.omcs.items()
        for mc_id, mc in mcs:
            dist = utils.distance_calc(mc.mean, pos, self.distance)
            if dist < min_dist:
                closest_mc = mc
                min_dist = dist
        return closest_mc

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs

    def make_mc(self, instance, timestamp, mc_id=None):
        if mc_id is None:
            mc_id = self.mc_id
        assert mc_id not in self.omcs.keys()
        assert mc_id not in self.pmcs.keys()
        # only one instance -> full weight
        cf1, cf2, w = instance_to_denstream_features(instance)
        mc = DenStreamMicroCluster(mc_id, cf1, cf2, w, timestamp, epsilon=self.epsilon, beta=self.beta, mu=self.mu,
                                   l=self.l, b=self.b)
        self.mc_id += 1
        return mc

    def get_summary(self):
        return {"pmcs": deepcopy(self.pmcs), "omcs": deepcopy(self.omcs)}


class DenStreamMicroCluster:
    def __init__(self, mc_id, cf1, cf2, w, timestamp, epsilon, beta, mu, l, b):
        self.mc_id = mc_id
        self.cf1 = deepcopy(cf1)
        self.cf2 = deepcopy(cf2)
        self.timestamp = timestamp
        self.creation_timestamp = timestamp
        self.w = w
        self.xi = 1
        self.radius = 0

        # hyperparameters
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.l = l
        self.b = b
        self.status = None
        #print(self.beta, self.mu, self.l, self.b, self.epsilon)
        self.fading_time = math.ceil(1 / l * math.log((beta * mu / (beta * mu - 1)), b))

        #helper
        self.check_time = timestamp
        self.calc_status(self.timestamp)

        #metrics
        #print(self.cf1)
        self.mean = np.array(self.cf1) / self.w
        self.radius = 0

    def calc_status(self, timestamp):
        if timestamp > self.timestamp:
            dcf_1 = self.b ** (-self.l * (timestamp - self.timestamp))
            for d in range(len(self.cf1)):
                self.cf1[d] = dcf_1 * self.cf1[d]
                self.cf2[d] = dcf_1 * self.cf2[d]
            self.w = dcf_1 * self.w
            self.timestamp = timestamp
            self.calc_xi()
        if self.w >= self.mu:
            self.status = "dense"
        elif self.w >= self.mu * self.beta:
            self.status = "potential"
        else:
            self.status = "outlier"
        self.check_time = timestamp

    def get_dict(self):
        dic = {}
        dic['mc_id'] = self.mc_id
        dic['cf1'] = self.cf1
        dic['cf2'] = self.cf2
        dic['w'] = self.w
        dic['timestamp'] = self.timestamp
        dic['creation_timestamp'] = self.creation_timestamp
        dic['radius'] = self.radius
        dic['mean'] = self.mean
        dic['epsilon'] = self.epsilon
        dic['beta'] = self.beta
        dic['mu'] = self.mu
        dic['l'] = self.l
        dic['b'] = self.b
        dic['status'] = self.status
        dic['xi'] = self.xi
        return dic

    def potentially_insert(self, instance, timestamp2):
        max_ts = max(self.timestamp, timestamp2)
        dcf_1 = self.b ** (-self.l * (max_ts - self.timestamp))
        dcf_2 = self.b ** (-self.l * (max_ts - timestamp2))
        cf12, cf22, w2 = instance_to_denstream_features(instance)
        cf1_test = []
        cf2_test = []
        #print(self.cf1, cf12)
        for d in range(len(cf12)):
            cf1_test.append(dcf_1 * self.cf1[d] + dcf_2 * cf12[d])
            cf2_test.append(dcf_1 * self.cf2[d] + dcf_2 * cf22[d])
        w_test = dcf_1 * self.w + dcf_2 * w2

        radius_test = calc_radius(cf1_test, cf2_test, w_test)
        if radius_test <= self.epsilon:
            self.cf1 = cf1_test
            self.cf2 = cf2_test
            self.w = w_test
            self.timestamp = max_ts
            self.radius = radius_test
            self.calc_xi()
            self.mean = np.array(self.cf1) / self.w
            self.calc_status(max_ts)
            return True
        else:
            return False

    def insert(self, cf12, cf22, w2, timestamp2):
        max_ts = max(self.timestamp, timestamp2)
        dcf_1 = self.b ** (-self.l * (max_ts - self.timestamp))
        dcf_2 = self.b ** (-self.l * (max_ts - timestamp2))

        for d in range(len(cf12)):
            self.cf1[d] = dcf_1 * self.cf1[d] + dcf_2 * cf12[d]
            self.cf2[d] = dcf_1 * self.cf2[d] + dcf_2 * cf22[d]
        self.w = dcf_1 * self.w + dcf_2 * w2
        self.timestamp = max_ts

        self.calc_stats()
        self.calc_status(self.timestamp)

    def calc_stats(self):
        self.radius = calc_radius(self.cf1, self.cf2, self.w)
        self.mean = np.array(self.cf1) / self.w
        self.calc_xi()

    def calc_xi(self):
        self.xi = (self.b ** (-self.l * (self.timestamp - self.creation_timestamp + self.fading_time)) - 1) / (
                self.b ** (-self.l * self.fading_time) - 1)

    def __str__(self):
        return str(self.get_dict())


def calc_radius(cf1, cf2, w):
    # stdev = ((np.array(cf2) / w) - (np.array(cf1) / w) ** 2) ** 0.5
    # radius = np.max(stdev)
    # return radius

    norm_cf1 = np.sqrt(np.sum(np.square(cf1)))
    norm_cf2 = np.sqrt(np.sum(np.square(cf2)))
    cf1_cf2_diff = norm_cf2 / w - (norm_cf1 / w) ** 2
    # print(cf1_cf2_diff)
    if cf1_cf2_diff > 0:
        radius = math.sqrt(cf1_cf2_diff)
    else:
        radius = 0
    return radius


def instance_to_denstream_features(instance):
    cf1 = []
    cf2 = []
    w = 1
    for d in range(len(instance)):
        cf1.append(instance[d])
        cf2.append(instance[d] ** 2)
    return cf1, cf2, w


import matplotlib.pyplot as plt
import matplotlib.patches as ptc

if __name__ == "__main__":
    stream, labels = load_data("complex9")
    ts = list(range(len(stream)))
    print(stream.shape)
    denstream = DenStream(epsilon=0.05, mu=20, init_num=100, heuristic=False, alpha=0.5, stream_speed=100)
    # assign, report = denstream.learn(stream, ts)
    # pmcs, omcs = denstream.get_summary()
    # print(len(pmcs), len(omcs))
    # for pmc_id, pmc in pmcs.items():
    #     pmc.calc_status(pmc.timestamp)
    #     print(pmc)
    # for omc_id, omc in omcs.items():
    #     omc.calc_status(omc.timestamp)
    #     print(omc)
    # print(report)
    make_fig = True
    for i in range(len(stream)):
        nmc_id, event_log = denstream.learn_one(stream[i], ts[i])
        print(i, event_log)
        summaries = denstream.get_summary()
        pmcs = summaries["pmcs"]
        omcs = summaries["omcs"]
        if i > 100 and make_fig:
            insert_id = None
            if "insert" in event_log:
                event_log_split = event_log.split(";")
                for event in event_log_split:
                    if "insert" in event:
                        insert_id = event.split(" ")[1]

            plt.figure(figsize=(8, 8))
            batch_start = max(0, i + 1- 500)
            for mc_id, mc in pmcs.items():
                #print(mc.mean)
                mc_patch = ptc.Circle((float(mc.mean[0]), float(mc.mean[1])), mc.radius, alpha=0.4, color="lightgrey")
                plt.scatter(x=mc.mean[0], y=mc.mean[1], c="orange", alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                if insert_id is not None:
                    if int(insert_id[1:]) == int(mc_id):
                        plt.scatter(x=mc.mean[0], y=mc.mean[1], c="pink", alpha=1, marker="x", s=150)
            for mc_id, mc in omcs.items():
                #print(mc.mean)
                mc_patch = ptc.Circle((float(mc.mean[0]), float(mc.mean[1])), mc.radius, alpha=0.4, color="darkgrey")
                plt.scatter(x=mc.mean[0], y=mc.mean[1], c="blue", alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                if insert_id is not None:
                    if int(insert_id[1:]) == int(mc_id):
                        plt.scatter(x=mc.mean[0], y=mc.mean[1], c="pink", alpha=1, marker="x", s=150)

            plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
                        cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
            plt.scatter(x=np.array(stream)[i:i + 1][:, 0], y=np.array(stream)[i:i + 1][:, 1], c="pink", s=25)
            plt.savefig(f"test_figures/denstream/complex9_denstream_{i}.png", bbox_inches='tight')
            plt.close()
