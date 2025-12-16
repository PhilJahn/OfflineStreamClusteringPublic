import math
from copy import deepcopy, copy
from time import sleep

from sklearn.neighbors import NearestNeighbors

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer
import numpy as np


class HDDStream(StreamClusterer):

    def __init__(self, init_num: int = 2000, window: int = 1000, epsilon=0.2, delta=0.001, pi=30, kappa=1000, beta=0.5, mu=10, l=0.5, b=2, verbose=False):
        super().__init__()
        self.init_num = init_num
        self.epsilon = epsilon
        self.delta = delta
        self.pi = pi
        self.kappa = kappa
        self.beta = beta
        self.mu = mu
        self.l = l
        self.b = b

        self.omcs = {}
        self.pmcs = {}
        self.mc_id = 0

        if not (1 >= beta > 0):
            raise ValueError("beta must be between 0 and 1")
        if not (mu * beta > 1):
            raise ValueError("mu * beta must be more than 1")

        self.initialized = False
        self.init_storage = []
        #self.init_timestamps = []

        self.window = window
        self.d_ctr = 0

        self.fading_time = math.ceil(1 / l * math.log((beta * mu / (beta * mu - 1)), b))

    def initialization(self):
        _, eps_nns = NearestNeighbors(radius=self.epsilon).fit(self.init_storage).radius_neighbors(
            self.init_storage, radius=self.epsilon)
        # print(eps_nns)
        indices = list(range(len(self.init_storage)))
        eps_nns_list = []
        self.init_storage = np.array(self.init_storage)
        for i in range(len(eps_nns)):
            eps_nns_list.append(sorted(eps_nns[i].tolist()))
        for i in range(len(eps_nns)):
            if i in indices:
                eps_nn = eps_nns_list[i]
                eps_nn_instances = self.init_storage[eps_nn]
                #print(eps_nn_instances)
                cf2 = np.sum(eps_nn_instances ** 2, axis=0)
                cf1 = np.sum(eps_nn_instances, axis=0)
                w = len(eps_nn_instances)
                pref_dims = calc_preferred_dims(cf1, cf2, w, self.delta)
                dim_pref_vector = calc_preferred_dim_vector(pref_dims, len(cf1), self.kappa)
                pref_eps_nn = []
                instance1 = self.init_storage[i]
                for j in indices:
                    instance2 = self.init_storage[j]
                    pref_dist = projected_distance(instance1, instance2, dim_pref_vector)
                    if pref_dist <= self.epsilon:
                        pref_eps_nn.append(j)
                cf2_proj = np.sum(self.init_storage[pref_eps_nn] ** 2, axis=0)
                cf1_proj = np.sum(self.init_storage[pref_eps_nn], axis=0)
                w_proj = len(pref_eps_nn)
                pref_dims_proj = calc_preferred_dims(cf1_proj, cf2_proj, w_proj, self.delta)

                # print(len(pref_dims), len(pref_eps_nn))
                if len(pref_dims_proj) <= self.pi and w_proj >= self.beta * self.mu:
                    mc = self.make_mc(instance1, self.t) # implied that inital all weighted same
                    for j in pref_eps_nn:
                        if j != i:
                            instance2 = self.init_storage[j]
                            cf12, cf22, w2 = instance_to_hddstream_features(instance2)
                            mc.insert(cf12, cf22, w2, self.t)
                        indices.remove(j)
                        for h in copy(eps_nns_list[j]):
                            eps_nns_list[h].remove(j)
                            if h != j:
                                eps_nns_list[j].remove(h)
                    # print(mc)
                    # print(mc.cf1, cf1, cf1_proj)
                    # print(mc.cf2, cf2, cf2_proj)
                    # print(pref_dims, len(pref_dims))
                    # print(pref_dims_proj, len(pref_dims_proj))
                    # print(mc.preferred_dims, len(mc.preferred_dims))
                    assert mc.status != "outlier"
                    if mc.radius <= self.epsilon: #skip anything that violates the epsilon criterion
                        #print(mc)
                        self.pmcs[mc.mc_id] = mc
        self.initialized = True

    def learn_one(self, x, t: int = None):
        if self.window is not None:
            self.t = self.d_ctr // self.window
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        event_log = ""
        x = np.array(x)
        if not self.initialized:
            if len(self.init_storage) < self.init_num:
                self.init_storage.append(x)
                #self.init_timestamps.append(self.t)
            if len(self.init_storage) == self.init_num:
                self.initialization()
                self.initialized = True
                return -1, "initialized;"
            else:
                return -1, "initialization;"
        else:
            cf1, cf2, w = instance_to_hddstream_features(x)
            new_id = None
            min_dist = np.inf
            min_valid = False
            min_id = None
            for pmc_id, pmc in self.pmcs.items():
                valid, dist = pmc.check_potential(cf1, cf2, w, self.t)
                if dist < min_dist:
                    min_dist = dist
                    min_valid = valid
                    min_id = pmc_id
            if min_valid:
                self.pmcs[min_id].insert(cf1, cf2, w, self.t)
                new_id = min_id
                event_log += "insert P" + str(min_id) + ";"
            else:
                for omc_id, omc in self.omcs.items():
                    valid, dist = omc.check_potential(cf1, cf2, w, self.t)
                    if dist < min_dist:
                        min_dist = dist
                        min_valid = valid
                        min_id = omc_id
                if min_valid:
                    self.omcs[min_id].insert(cf1, cf2, w, self.t)
                    new_id = min_id
                    event_log += "insert O" + str(min_id) + ";"
                    if self.omcs[min_id] != "outlier" and self.omcs[min_id] != "?":
                        self.pmcs[min_id] = self.omcs.pop(min_id)
                        event_log += "upgrade O" + str(min_id) + ";"
                else:
                    new_mc = self.make_mc(x, self.t)
                    new_id = new_mc.mc_id
                    self.omcs[new_id] = new_mc
                    event_log += "new O" + str(new_id) + ";"

            if self.t % self.fading_time == 0:
                if self.window is None or self.d_ctr % self.window == 0:
                    print("check time", self.t, self.d_ctr)
                    down_pmc_ids = []
                    for pmc_id, pmc in self.pmcs.items():
                        pmc.calc_status(self.t)
                        print(pmc)
                        if pmc.status == "outlier":
                            down_pmc_ids.append(pmc_id)
                            event_log += "downgrade P" + str(pmc_id) + ";"
                    for down_pmc_id in down_pmc_ids:
                        self.omcs[down_pmc_id] = self.pmcs.pop(down_pmc_id)
                    del_omc_ids = []
                    for omc_id, omc in self.omcs.items():
                        omc.calc_status(self.t)
                        print(omc)
                        if omc.status == "outlier":
                            if omc.wexp > omc.w:
                                del_omc_ids.append(omc_id)
                                event_log += "delete O" + str(omc_id) + ";"
                        else:
                            del_omc_ids.append(omc_id)
                            self.pmcs[omc_id] = omc
                            event_log += "upgrade O" + str(omc_id) + ";"
                    for del_omc_id in del_omc_ids:
                        self.omcs.pop(del_omc_id)

            return new_id, event_log

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
        cf1, cf2, w = instance_to_hddstream_features(instance)
        mc = HDDStreamMicroCluster(mc_id, cf1, cf2, w, timestamp, epsilon=self.epsilon, delta=self.delta,
                                   kappa=self.kappa, pi=self.pi, beta=self.beta, mu=self.mu, l=self.l, b=self.b)
        self.mc_id += 1
        return mc

    def get_summary(self):
        return {"pmcs": deepcopy(self.pmcs), "omcs": deepcopy(self.omcs)}

class HDDStreamMicroCluster:
    def __init__(self, mc_id, cf1, cf2, w, timestamp, epsilon, delta, kappa, pi, beta, mu, b, l):
        self.mc_id = mc_id
        self.cf1 = deepcopy(cf1)
        self.cf2 = deepcopy(cf2)
        self.w = w
        self.timestamp = timestamp
        self.creation_timestamp = timestamp

        #hyperparameters
        self.epsilon = epsilon
        self.delta = delta
        self.kappa = kappa
        self.pi = pi
        self.beta = beta
        self.mu = mu
        self.b = b
        self.l = l
        self.fading_time = math.ceil(1 / l * math.log((beta * mu / (beta * mu - 1)), b))

        #metrics
        self.preferred_dims = None
        self.dim_pref_vector = None
        self.status = None
        self.radius = None
        self.mean = None
        self.wexp = None
        self.calc_status(self.timestamp)

    def check_potential(self, cf12, cf22, w2, timestamp):
        self.apply_decay(timestamp)
        temp_cf2 = np.array(self.cf2) + cf22
        temp_cf1 = np.array(self.cf1) + cf12
        temp_w = self.w + w2
        temp_preferred_dims = calc_preferred_dims(temp_cf1, temp_cf2, temp_w, self.delta)
        if len(temp_preferred_dims) > self.pi and not self.status == "outlier":
            return False, np.inf
        temp_dim_pref_vector = calc_preferred_dim_vector(temp_preferred_dims, len(temp_cf1), self.kappa)
        pos = np.array(cf12) / w2
        #print(pos, self.mean, temp_dim_pref_vector)
        dist = projected_distance(pos, self.mean, temp_dim_pref_vector)
        radius = calc_radius(temp_cf1, temp_cf2, temp_w, temp_dim_pref_vector)
        if radius > self.epsilon:
            return False, dist
        else:
            return True, dist

    def insert(self, cf12, cf22, w2, timestamp2):
        max_ts = max(self.timestamp, timestamp2)
        dcf_1 = self.b ** (-self.l * (max_ts - self.timestamp))
        dcf_2 = self.b ** (-self.l * (max_ts - timestamp2))

        for d in range(len(cf12)):
            self.cf1[d] = dcf_1 * self.cf1[d] + dcf_2 * cf12[d]
            self.cf2[d] = dcf_1 * self.cf2[d] + dcf_2 * cf22[d]
        self.w = dcf_1 * self.w + dcf_2 * w2
        self.timestamp = max_ts
        self.calc_status(self.timestamp)

    def calc_status(self, timestamp):
        self.apply_decay(timestamp)
        self.preferred_dims = calc_preferred_dims(self.cf1, self.cf2, self.w, self.delta)
        self.dim_pref_vector = calc_preferred_dim_vector(self.preferred_dims, len(self.cf1), self.kappa)
        self.wexp = (self.b ** (-self.l * (timestamp- self.creation_timestamp + self.fading_time)) - 1) / (
                self.b ** (-self.l * self.fading_time) - 1)
        self.radius = calc_radius(self.cf1, self.cf2, self.w, self.dim_pref_vector)
        self.mean = np.array(self.cf1)/self.w
        if self.radius > self.epsilon:
            self.status = "?"
        elif self.w < self.mu * self.beta or len(self.preferred_dims) > self.pi:
            self.status = "outlier"
        elif self.w >= self.mu:
            self.status = "dense"
        else:
            assert self.w >= self.mu * self.beta
            self.status = "potential"

    def apply_decay(self, timestamp):
        if timestamp > self.timestamp:
            dcf_1 = self.b ** (-self.l * (timestamp - self.timestamp))
            for d in range(len(self.cf1)):
                self.cf1[d] = dcf_1 * self.cf1[d]
                self.cf2[d] = dcf_1 * self.cf2[d]
            self.w = dcf_1 * self.w
            self.timestamp = timestamp

    def get_dict(self):
        dic = {}
        dic['mc_id'] = self.mc_id
        dic['cf1'] = self.cf1
        dic['cf2'] = self.cf2
        dic['w'] = self.w
        dic['timestamp'] = self.timestamp
        dic['creation_timestamp'] = self.creation_timestamp
        dic['preferred_dims'] = self.preferred_dims
        dic['dim_pref_vector'] = self.dim_pref_vector
        dic['radius'] = self.radius
        dic['mean'] = self.mean
        dic['epsilon'] = self.epsilon
        dic['beta'] = self.beta
        dic['mu'] = self.mu
        dic['l'] = self.l
        dic['b'] = self.b
        dic['status'] = self.status
        dic['wexp'] = self.wexp
        dic['var'] = np.sqrt((np.array(self.cf2) / self.w) - (np.array(self.cf1) / self.w) ** 2)
        return dic

    def __str__(self):
        return str(self.get_dict())


def calc_preferred_dims(cf1, cf2, w, delta):
    var = np.sqrt((np.array(cf2) / w) - (np.array(cf1) / w) ** 2)
    preferred_dims = [d for d in range(len(var)) if var[d] <= delta]
    #print(var, preferred_dims)
    return preferred_dims


def calc_preferred_dim_vector(preferred_dims, dims, kappa):
    return np.array([kappa if d in preferred_dims else 1 for d in range(dims)])


def calc_radius(cf1, cf2, w, dim_pref_vector):
    std = (np.array(cf2) / w) - (np.array(cf1) / w) ** 2
    radius = np.sqrt(np.sum((1/dim_pref_vector) * std))
    return radius


def projected_distance(pos1, pos2, dim_pref_vector):
    return np.sqrt(np.sum((1 / dim_pref_vector) * np.square(pos1 - pos2)))


def instance_to_hddstream_features(instance):
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
    make_fig = True
    #{'b': 2, 'batch_budget': 1800, 'batch_size': 1000, 'beta': 0.4036270526434, 'delta': 0.0082182551841,
    # 'ds': 'densired10', 'epsilon': 0.0749471238297, 'init_num': 1000, 'kappa': 5, 'l': 0.1743064853777,
    # 'mu': 9, 'offline': None, 'online': 'hddstream', 'pi': 5, 'window': 1000}
    stream, labels = load_data("densired10")  #[[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    hddstream = HDDStream(epsilon=0.0749471238297, l=0.1743064853777, kappa=5, b=2, mu=9, init_num=1000, pi=5, window=200,
                          delta=0.0082182551841, beta= 0.4036270526434
                          )
    for i in range(len(stream)):
        nmc_id, event_log = hddstream.learn_one(stream[i], ts[i])
        print(i, event_log)
        summary = hddstream.get_summary()
        pmcs = summary["pmcs"]
        omcs = summary["omcs"]
        if i > 100 and make_fig:
            insert_id = None
            if "insert" in event_log:
                event_log_split = event_log.split(";")
                for event in event_log_split:
                    if "insert" in event:
                        insert_id = event.split(" ")[1]

            plt.figure(figsize=(8, 8))
            batch_start = max(0, i + 1 - 500)
            for mc_id, mc in pmcs.items():
                # print(mc.mean)
                mc_patch = ptc.Circle((float(mc.mean[0]), float(mc.mean[1])), mc.radius, alpha=0.4, color="lightgrey")
                plt.scatter(x=mc.mean[0], y=mc.mean[1], c="orange", alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                if insert_id is not None:
                    if int(insert_id[1:]) == int(mc_id):
                        plt.scatter(x=mc.mean[0], y=mc.mean[1], c="pink", alpha=1, marker="x", s=150)
            for mc_id, mc in omcs.items():
                # print(mc.mean)
                mc_patch = ptc.Circle((float(mc.mean[0]), float(mc.mean[1])), mc.radius, alpha=0.4, color="darkgrey")
                plt.scatter(x=mc.mean[0], y=mc.mean[1], c="blue", alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                if insert_id is not None:
                    if int(insert_id[1:]) == int(mc_id):
                        plt.scatter(x=mc.mean[0], y=mc.mean[1], c="pink", alpha=1, marker="x", s=150)

            plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
                        cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
            plt.scatter(x=np.array(stream)[i:i + 1][:, 0], y=np.array(stream)[i:i + 1][:, 1], c="pink", s=25)
            # plt.savefig(f"test_figures/hddstream/complex9_hddstream_{i}.png", bbox_inches='tight')
            plt.close()

    # pos1 = np.array([1,24])
    # pos2 = np.array([2,112])
    # dim_pref_vector = np.array([10,10])
    # dist1 = projected_distance(pos1, pos2, dim_pref_vector)
    # dist2 = utils.distance_calc(pos1, pos2, "euclidean")
    # print(dist1, dist2)
