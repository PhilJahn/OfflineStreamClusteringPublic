from copy import deepcopy

import numpy as np
from scipy.cluster.hierarchy import weighted
from sklearn.cluster import KMeans

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer
from numpy.random import PCG64

class STREAMKMPlus(StreamClusterer):
    # m: size of coreset
    def __init__(self, m, random_state=42):
        super().__init__()
        self.m = m
        self.buckets = [[]]
        self.random_generator = np.random.Generator(PCG64(random_state))

    def learn_one(self, x, t: int = None):
        event_log = ""
        x = CoreSetPoint(np.array(x), 1)
        if len(self.buckets[0]) < self.m:
            self.buckets[0].append(x)
            if len(self.buckets[0]) == self.m:
                i = 1
                while i < len(self.buckets) and len(self.buckets[i]) > 0:
                    self.buckets[i].extend(self.buckets[i-1])
                    #print("len", len(self.buckets[i]))
                    coreset = self.get_coreset(self.buckets[i])
                    self.buckets[i-1] = []
                    self.buckets[i] = coreset
                    i += 1
                if i == len(self.buckets):
                    self.buckets.append([])
                self.buckets[i] = self.buckets[i-1]
                self.buckets[i-1] = []
                event_log += f"coreset {i};"
        else:
            raise Exception("Bucket 0 is too large")

        return -1, event_log

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs

    def get_buckets(self):
        return self.buckets

    def get_summary(self):
        all_coreset_points = []
        for bucket in self.buckets:
            all_coreset_points.extend(bucket)
        summary = deepcopy(self.get_coreset(all_coreset_points))
        return dict(zip(list(range(len(summary))), summary))

    def get_coreset(self, points):
        q1 = self.random_generator.choice(points)
        #print(q1)
        positions = [point.position for point in points]
        dists = utils.distance_calc(q1.position, positions, "euclidean")
        dist_weight = np.sum(dists**2)
        root = CoreSetTreeNode(None, q1, points, dist_weight)
        coreset_size = 1
        while coreset_size < self.m and root.weight > 0:
            node = root
            #lvl = 0
            while not node.is_leaf():
                #print(lvl)
                rand_val = self.random_generator.random()
                node = node.get_child(rand_val)
                #lvl += 1
            #print("at leaf")
            #print("node", node)
            child_node = False
            while not child_node:
                rand_val2 = self.random_generator.random()
                child_node = node.split(rand_val2)
            coreset_size += 1
            #print("tree", root)
        coreset_leaves = root.get_all_leafs()
        #print(coreset_leaves)
        coreset = []
        cum_weight = 0
        for coreset_leaf in coreset_leaves:
            coreset.append(CoreSetPoint(coreset_leaf.sample_point.position, coreset_leaf.data_weight))
            cum_weight += coreset_leaf.data_weight
        return coreset




class CoreSetTreeNode():
    def __init__(self, parent, sample_point, data_points, dist_weight):
        self.parent = parent
        self.child1 = None
        self.child2 = None
        self.data_points = np.array(data_points)
        self.sample_point = sample_point
        self.dist_weight = dist_weight
        self.data_weight = np.sum([data_point.weight for data_point in data_points])
        self.weight = self.dist_weight * self.data_weight
        self.size = len(data_points)

    def is_leaf(self):
        return self.child1 is None

    def get_child(self, rand_val):
        #print(self.weight, self.child1.weight, self.child2.weight)
        if self.child1.weight == 0:
            #print("child2", self.child2.weight)
            return self.child2
        elif self.child2.weight == 0:
            #print("child1",self.child1.weight)
            return self.child1
        else:
            if self.child1.weight/self.weight < rand_val:
                return self.child2
            else:
                return self.child1

    def get_all_leafs(self):
        if not self.is_leaf():
            leaves = self.child1.get_all_leafs()
            leaves.extend(self.child2.get_all_leafs())
            return leaves
        else:
            return [self]

    def split(self, rand_val):

        positions = [point.position for point in self.data_points]
        dists_org = utils.distance_calc(self.sample_point.position, positions, "euclidean")
        data_weights = [data_point.weight for data_point in self.data_points]
        data_weights = np.array(data_weights)
        #print(self.weight, sum(dists_org))
        if self.weight == 0:
            raise Exception

        weighted_dists_org = dists_org * data_weights
        sample_weights = weighted_dists_org/sum(weighted_dists_org)
        cumulative_weights = np.cumsum(sample_weights)
        #print(sample_weights)
        #print(cumulative_weights)
        #assert(np.isclose(sum(sample_weights), 1))
        chosen = np.searchsorted(cumulative_weights, rand_val, side='left')
        if dists_org[chosen] == 0:
            return False
        sample_point_new = self.data_points[chosen]
        #print(sample_point_new)
        dists_new = utils.distance_calc(sample_point_new.position, positions, "euclidean")
        closer_org = np.where(dists_org - dists_new <= 0)[0]
        closer_new = np.where(dists_org - dists_new > 0)[0]
        #print(closer_org, closer_new)
        assert(len(closer_org)+len(closer_new)==self.size)
        data_points_org = self.data_points[closer_org]
        dist_weight_org = np.sum(dists_org[closer_org]**2)
        data_points_new = self.data_points[closer_new]
        dist_weight_new = np.sum(dists_new[closer_new]**2)

        self.child1 = CoreSetTreeNode(self, self.sample_point, data_points_org, dist_weight_org)
        self.child2 = CoreSetTreeNode(self, sample_point_new, data_points_new, dist_weight_new)
        self.data_points = None
        self.propagate_weights()

        return self.child2

    def propagate_weights(self):
        assert not self.is_leaf()
        self.weight = self.child1.weight + self.child2.weight
        if self.parent is not None:
            self.parent.propagate_weights()

    def __str__(self):
        return f"[{self.sample_point}:{self.weight},{self.size}: {self.child1} {self.child2}]"

class CoreSetPoint:
    def __init__(self, position, weight):
        self.position = position
        self.weight = weight

# def cost(sample, ps):
#     return np.sum(np.sqrt(np.sum(np.square(sample - ps), axis=-1)))


import matplotlib.pyplot as plt

if __name__ == "__main__":
    stream, labels = load_data("stream_densired")
    ts = list(range(len(stream)))
    print(stream.shape)
    data = stream
    lb = labels
    streamkm = STREAMKMPlus(m=100)

    # coreset = np.array(streamkm.get_coreset(data))
    # plt.figure(figsize=(8, 8))
    # plt.scatter(x=np.array(data)[:, 0], y=np.array(data)[:, 1],
    #             cmap=plt.get_cmap("jet"), c=lb, s=10)
    # plt.scatter(x=coreset[:,0], y=coreset[:,1], c="black", alpha=1, s=100)
    # plt.show()

    make_fig = False
    for i in range(len(stream)):
        print("i", i)
        nmc_id, event_log = streamkm.learn_one(stream[i], ts[i])
        buckets = streamkm.get_buckets()
        corepoints = streamkm.get_summary()

        # if ";" in event_log:
        #     print(i, event_log)
        #     #print(buckets)
        #     plt.figure(figsize=(8, 8))
        #     plt.scatter(x=np.array(stream)[:i + 1][:, 0], y=np.array(stream)[:i + 1][:, 1],
        #                 cmap=plt.get_cmap("jet"), c=labels[:i + 1], s=10)
        #     for j in range(len(buckets)):
        #         if len(buckets[j]) > 0:
        #             plt.scatter(x=np.array(buckets[j])[:, 0], y=np.array(buckets[j])[:, 1], c="black", alpha=1, s=10*j)
        #     plt.scatter(x=np.array(corepoints)[:, 0], y=np.array(corepoints)[:, 1], c="grey", alpha=1, s=30)
        #     plt.ylim((0, 1))
        #     plt.xlim((0, 1))
        #     plt.savefig(f"test_figures/streamkmplus/complex9_streamkmplus_{i}.png", bbox_inches='tight')
        #     plt.close()

    # sample = np.array([1,1])
    # allpoints = np.array([[1,1], [2,1], [3,5], [1,2], [1.5, 0.9]])
    # weight = np.sum(utils.distance_calc(sample, allpoints, "euclidean"))
    # ctnode = CoreSetTreeNode(None, sample, allpoints, weight)
    #
    # ctnode.split(0.93)
    #
    # print(ctnode)
    #
    # weight = np.sum(utils.distance_calc(sample, allpoints[[0,3,4]], "euclidean"))
    # print(weight)
    #
    # weight = np.sum(utils.distance_calc(allpoints[2], allpoints[[1,2]], "euclidean"))
    # print(weight)
