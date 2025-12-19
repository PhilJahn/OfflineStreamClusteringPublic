import copy
import math

import numpy as np

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer


class SWClustering(StreamClusterer):

    def __init__(self, epsilon, N):
        super().__init__()
        self.epsilon = epsilon
        self.N = N


class EHCF:
    def __init__(self, instance, t, epsilon, N):
        initial_tcf = make_tcf(instance, t)
        self.tcfs = {0: [initial_tcf]}
        self.t = t
        self.cf1x = copy.deepcopy(initial_tcf.cf1x)
        self.cf2x = copy.deepcopy(initial_tcf.cf2x)
        self.n = 1

        self.N = N
        self.max_per_level = math.ceil(1.0/epsilon) + 2

    def update(self, instance, t):
        i = 0
        self.t = max(t, self.t)
        cur_tcf = make_tcf(instance, t)
        self.cf1x += cur_tcf.cf1x
        self.cf2x += cur_tcf.cf2x
        self.n += cur_tcf.n
        while len(self.tcfs[i]) == self.max_per_level-1:
            #merge
            self.tcfs[i].append(cur_tcf)
            cur_tcf = self.tcfs[i].pop(0)
            cur_tcf.insert(self.tcfs[i].pop(0))
            assert len(self.tcfs[i]) == self.max_per_level-2
            i += 1
            if i not in self.tcfs.keys():
                self.tcfs[i] = []



        self.tcfs[i].append(cur_tcf)

        highest_level = max(self.tcfs.keys())
        oldest = self.tcfs[highest_level][-1]
        if oldest.t < self.t - self.N:
            self.tcfs[highest_level].pop(-1)
            self.cf1x -= oldest.cf1x
            self.cf2x -= oldest.cf2x
            self.n -= oldest.n



        sum_cf1x = np.zeros(len(self.cf1x))
        sum_cf2x = np.zeros(len(self.cf2x))
        sum_n = 0
        for level in self.tcfs.values():
            for tcf in level:
                sum_cf1x += tcf.cf1x
                sum_cf2x += tcf.cf2x
                sum_n += tcf.n
        assert self.cf1x == sum_cf1x

    def get_dict(self):
        dic = {}
        dic["cf1x"] = self.cf1x
        dic["cf2x"] = self.cf2x
        dic["n"] = self.n
        dic["t"] = self.t
        tcfs = {}
        for key, level in self.tcfs.items():
            tcfs[key] = [str(tcf) for tcf in level]
        dic["tcfs"] = tcfs
        return dic

    def __str__(self):
        return str(self.get_dict())

class TCF:
    def __init__(self, cf1x, cf2x, t, n):
        self.cf1x = cf1x
        self.cf2x = cf2x
        self.t = t
        self.n = n

    def insert_instance(self, instance, t):
        make_tcf(instance, t)
        self.insert(instance)

    def insert(self, tcf):
        self.t = max(self.t, tcf.t)
        self.n = self.n + tcf.n
        self.cf1x = self.cf1x + tcf.cf1x
        self.cf2x = self.cf2x + tcf.cf2x

    def get_dict(self):
        dic = {}
        dic["cf1x"] = self.cf1x
        dic["cf2x"] = self.cf2x
        dic["n"] = self.n
        dic["t"] = self.t
        return dic

    def __str__(self):
        return str(self.get_dict())



def make_tcf(instance, t):
    cf2x = []
    cf1x = []
    for d in range(len(instance)):
        cf2x.append(float(instance[d]) ** 2)
        cf1x.append(float(instance[d]))
    return TCF(np.array(cf1x), np.array(cf2x), t, 1)


import matplotlib.pyplot as plt
import matplotlib.patches as ptc
if __name__ == "__main__":
    stream, labels = load_data("complex9")  #[[0],[1],[3],[7],[8],[9]]
    stream = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[12],[13],[14],[15],[16], [17], [2.5], [2.6]]
    ts = list(range(len(stream)))
    make_fig = True


    ehcf = EHCF(stream[0], 0, 2, 10)
    print(ehcf)
    for i in range(1,len(stream)):
        ehcf.update(stream[i], i)
        print(ehcf)

    raise Exception
    swclustering = SWClustering()
    for i in range(len(stream)):
        print("TS", i, "started")
        nmc_id, event_log = betula.learn_one(stream[i], ts[i])
        print("TS", i, "done")
        if make_fig and ("recon" in event_log or (i+1) % 100 == 0):
            entries, outliers, tree = betula.get_summary()
            #print("entries", len(entries))
            #print("tree", tree)
            #print("outliers", outliers)
            entry_parents = [str([entry.parent]).split(" ")[-1][:-2] for entry in entries]
            entry_parents_unique = np.unique(sorted(entry_parents)).tolist()

            plt.figure(figsize=(8,8))
            batch_start = max(0, i + 1 - 500)
            for entry in entries:
                parent_id = entry_parents_unique.index(str([entry.parent]).split(" ")[-1][:-2])
                mc_patch = ptc.Circle((float(entry.mean[0]), float(entry.mean[1])), entry.radius, alpha=0.4, color="lightgrey")
                plt.scatter(x=entry.mean[0], y=entry.mean[1], c=parent_id, alpha=1, s=100)
                plt.gca().add_patch(mc_patch)
                plt.text(entry.mean[0]+0.02, entry.mean[1], s=parent_id)
                #plt.text(entry.mean[0], entry.mean[1]-0.05, s=entry.mean)
            if "recon" in event_log:
                plt.title("Reconstruction")
            plt.scatter(x=np.array(stream)[batch_start:i+1][:,0], y=np.array(stream)[batch_start:i+1][:,1], cmap=plt.get_cmap("jet"), c=labels[batch_start:i+1], s=10)
            plt.scatter(x=np.array(stream)[i:i+1][:,0], y=np.array(stream)[i:i+1][:,1], c="pink", s=25)
            plt.savefig(f"test_figures/birch/complex9_birch_{i}.png", bbox_inches='tight')
            plt.close()
