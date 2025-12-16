import copy
import math
from pprint import pprint

from pandas._libs.tslibs import timestamps

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer

import numpy as np

class GBFuzzyStream(StreamClusterer):

    def __init__(self, l=0.2, threshold=0.3, m=2.0, eps=10, dist="euclidean", window=1000):
        super().__init__()
        self.l = l
        self.threshold = threshold
        self.m = m
        self.eps = eps
        self.dist = dist
        self.window = window
        self.d_ctr = 0

        self.fmbs = []
        self.cur_data = []
        self.cur_sample_ids = []

        self.initialized = False


    def split_by_dm(self, balls):
        split_balls = []
        while len(balls) > 0:
            new_balls = []
            for ball in balls:
                ball_new1, ball_new2, dm_new = ball.split()
                if dm_new > ball.dm:
                    new_balls.append(ball_new1)
                    new_balls.append(ball_new2)
                else:
                    split_balls.append(ball)
            balls = new_balls
        return split_balls

    def generate_fgbs(self):
        reducible_fgbs = [FuzzyGranularBall(self.cur_data, self.cur_sample_ids, self.dist, eps=self.eps, m=self.m)]
        initial_fgbs = self.split_by_dm(reducible_fgbs)
        radii = []
        for fgb in initial_fgbs:
            radii.append(fgb.radius)
        #print(initial_fgbs)
        radius_threshold = 1.5 * max(np.mean(radii), np.median(radii))
        #print(radius_threshold)
        final_fgbs = []
        while len(initial_fgbs) > 0:
            new_fgbs = []
            for fgb in initial_fgbs:
                if fgb.radius > radius_threshold:
                    fgb_new1, fgb_new2, dm_new = fgb.split()
                    if fgb_new1 is not None and fgb_new2 is not None:
                        new_fgbs.append(fgb_new1)
                        new_fgbs.append(fgb_new2)
                    else: # can't actually split further
                        final_fgbs.append(fgb)
                else:
                    final_fgbs.append(fgb)
            initial_fgbs = new_fgbs
        return final_fgbs

    def learn_one(self, x, t: int = None):
        if self.window is not None:
            self.t = self.d_ctr // self.window
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        event_log = ""
        x = np.array(x)
        self.cur_data.append(x)
        self.cur_sample_ids.append(self.d_ctr-1)

        if (self.d_ctr+1) % self.window == 1:
            self.cur_data = np.array(self.cur_data)
            self.cur_sample_ids = np.array(self.cur_sample_ids)
            #print(self.t, self.d_ctr, len(self.cur_data))
            fgbs = self.generate_fgbs()
            if not self.initialized:
                for fgb in fgbs:
                    fmb = FuzzyMicroBall(fgb.samples, fgb.sample_ids, np.array([self.t] * len(fgb.samples)), dist=self.dist, eps=self.eps, m=self.m, l= self.l)
                    self.fmbs.append(fmb)
                self.initialized = True
            else:
                #print(len(fgbs), len(self.cur_data))
                for fgb in fgbs:
                    fmb_centers = [fmb.center for fmb in self.fmbs]
                    if len(fmb_centers) >= 1:
                        fgb_dist = utils.distance_calc(fgb.center, np.array(fmb_centers), dist=self.dist)
                        #print(fgb_dist)
                        closest_fmb = np.argmin(fgb_dist)
                        closest_dist = fgb_dist[closest_fmb]
                    else:
                        closest_dist = np.inf

                    if len(fmb_centers) > 1:
                        fmb_dists = utils.distance_calc(fmb_centers[closest_fmb], np.array(fmb_centers), dist=self.dist)
                        nn_dist = np.partition(fmb_dists,1)[1]
                    else:
                        nn_dist = 0

                    if nn_dist < closest_dist + fgb.radius:
                        fmb = FuzzyMicroBall(fgb.samples, fgb.sample_ids, np.array([self.t] * len(fgb.samples)), dist=self.dist, eps=self.eps,
                                             m=self.m, l=self.l)
                        self.fmbs.append(fmb)
                    else:
                        self.fmbs[closest_fmb].merge(fgb, self.t)
                        split_fmbs = self.split_by_dm([self.fmbs[closest_fmb]])
                        #print("split_fmbs", split_fmbs)
                        #print("before", len(self.fmbs))
                        self.fmbs.remove(self.fmbs[closest_fmb])
                        #print("between", len(self.fmbs))
                        self.fmbs.extend(split_fmbs)
                        #print("after", len(self.fmbs))
            del_fmbs = []
            for fmb in self.fmbs:
                fmb.update_wav(self.t)
                if fmb.wav < self.threshold:
                    del_fmbs.append(fmb)
            for del_fmb in del_fmbs:
                self.fmbs.remove(del_fmb)
                event_log += f"del FMB"
            self.cur_data = []
            self.cur_sample_ids = []
        return -1, event_log

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs

    def get_summary(self):
        return dict(zip(list(range(len(self.fmbs))), copy.deepcopy(self.fmbs)))



class FuzzyGranularBall:
    def __init__(self, samples, sample_ids, dist, eps, m): # per definition only exists for single timestep
        self.samples = samples
        self.sample_ids = sample_ids
        self.dist = dist
        self.eps = eps
        self.m = m

        self.dm = None # dm
        self.center = None
        self.radius = None
        self.compute_metrics()

    def compute_metrics(self):
        self.center = np.mean(self.samples, axis=0)
        dists = utils.distance_calc(self.center, self.samples, dist=self.dist)
        #print(len(self.samples), len(np.unique(self.samples, axis=0)))
        self.radius = np.max(dists)
        self.dm = len(self.samples) / np.sum(dists ** 2)

    def make_children(self,assignment1, assignment2):
        samples1 = self.samples[assignment1]
        sample_ids1 = self.sample_ids[assignment1]
        fgb1 = FuzzyGranularBall(samples1, sample_ids1, dist=self.dist, eps=self.eps, m=self.m)
        samples2 = self.samples[assignment2]
        sample_ids2 = self.sample_ids[assignment2]
        fgb2 = FuzzyGranularBall(samples2, sample_ids2, dist=self.dist, eps=self.eps, m=self.m)
        return fgb1, fgb2

    def split(self):
        if len(self.samples) > 1:
            dists = utils.distance_calc(self.center, self.samples, dist=self.dist)
            p1 = np.argmax(dists)
            dists_p1 = utils.distance_calc(self.samples, self.samples[p1], dist=self.dist)
            p2 = np.argmax(dists_p1)

            o1 = (self.center + self.samples[p1]) / 2
            o2 = (self.center + self.samples[p2]) / 2
            #print(o1, o2)
            o1, o2, u_ball = self.iterate_child_centers(o1, o2)
            assignment = np.argmax(u_ball, axis=0)
            #print(assignment)
            assignment1 = np.where(assignment == 0)[0]
            assignment2 = np.where(assignment == 1)[0]

            if len(assignment1) > 0 and len(assignment2) > 0:
                child1, child2 = self.make_children(assignment1, assignment2)
                dm1 = child1.dm
                dm2 = child2.dm
                dm_child = (len(assignment1) / len(self.samples)) * dm1 + (len(assignment2) / len(self.samples)) * dm2
            else:
                #print("assign", assignment)
                return None, None, -np.inf
            if len(self.samples) > 7 and len(child1.samples) > 1 and len(child2.samples) > 1: # based on original implementation
                return child1, child2, dm_child
            else:
                #print("self", len(self.samples), "c1", len(child1.samples), "c2", len(child2.samples))
                return child1, child2, -np.inf
        else:
            #print("one")
            return None, None, -np.inf

    def iterate_child_centers(self, o1, o2):
        u_ball = self.compute_u(o1, o2)
        #print(u_ball)
        o1_new = np.sum((u_ball[0].reshape(-1, 1) ** self.m) * self.samples, axis=0) / np.sum(u_ball[0] ** self.m)
        o2_new = np.sum((u_ball[1].reshape(-1, 1) ** self.m) * self.samples, axis=0) / np.sum(u_ball[1] ** self.m)
        u_ball_new = self.compute_u(o1_new, o2_new)
        #print(o1_new, o2_new)
        #print(u_ball_new)
        while (np.sum(np.abs(u_ball - u_ball_new)) >= self.eps):
            u_ball = u_ball_new
            o1_new = np.sum((u_ball[0].reshape(-1, 1) ** self.m) * self.samples, axis=0) / np.sum(u_ball[0] ** self.m)
            o2_new = np.sum((u_ball[1].reshape(-1, 1) ** self.m) * self.samples, axis=0) / np.sum(u_ball[1] ** self.m)
            u_ball_new = self.compute_u(o1_new, o2_new)
            #print(o1_new, o2_new)
            #print(u_ball_new)
        return o1_new, o2_new, u_ball_new

    def compute_u(self, o1, o2):
        dists1 = utils.distance_calc(np.array(o1), self.samples, dist="euclidean")
        dists2 = utils.distance_calc(np.array(o2), self.samples, dist="euclidean")
        m_exp = 2 / (self.m - 1)
        div1 = (((dists1 ** 2) / (dists2 ** 2)) ** m_exp) + 1
        div2 = (((dists2 ** 2) / (dists1 ** 2)) ** m_exp) + 1
        u1 = 1 / div1
        u2 = 1 / div2
        return np.array([u1, u2])

    def get_dict(self):
        dic = {}
        dic["center"] = self.center
        dic["num_samples"] = len(self.samples)
        dic["sample_ids"] = self.sample_ids
        dic["samples"] = self.samples
        dic["radius"] = self.radius
        dic["dm"] = self.dm
        dic["dist"] = self.dist
        dic["eps"] = self.eps
        dic["m"] = self.m
        return dic

    def __str__(self):
        return str(self.get_dict())


class FuzzyMicroBall(FuzzyGranularBall):
    def __init__(self, samples, sample_ids, timestamps, dist, eps, m, l):
        super().__init__(samples, sample_ids, dist, eps, m)
        self.timestamps = timestamps
        self.wav = 1
        self.l = l

    def make_children(self,assignment1, assignment2):
        samples1 = self.samples[assignment1]
        sample_ids1 = self.sample_ids[assignment1]
        timestamps1 = self.timestamps[assignment1]
        fmb1 = FuzzyMicroBall(samples1, sample_ids1, timestamps1, dist=self.dist, eps=self.eps, m=self.m, l=self.l)

        samples2 = self.samples[assignment2]
        sample_ids2 = self.sample_ids[assignment2]
        timestamps2 = self.timestamps[assignment2]
        fmb2 = FuzzyMicroBall(samples2, sample_ids2, timestamps2, dist=self.dist, eps=self.eps, m=self.m, l=self.l)
        return fmb1, fmb2

    def merge(self, fgb, timestamp):
        samples_list = self.samples.tolist()
        samples_list.extend(fgb.samples)
        self.samples = np.array(samples_list)
        sample_ids_list = self.sample_ids.tolist()
        sample_ids_list.extend(fgb.sample_ids)
        self.sample_ids = np.array(sample_ids_list)
        timestamps_list = self.timestamps.tolist()
        timestamps_list.extend([timestamp]*len(fgb.samples))
        self.timestamps = np.array(timestamps_list)
        self.compute_metrics()
        
    def update_wav(self, timestamp):
        timestamp_vals, timestamp_nums = np.unique(np.array(self.timestamps), return_counts=True)
        assert sum(timestamp_nums) == len(self.samples)
        weights = [(timestamp_nums[i]/len(self.samples)) * (2.0**(-1*self.l*(timestamp-timestamp_vals[i]))) for i in range(len(timestamp_nums))]
        #print(timestamp_nums, timestamp_vals, weights)
        self.wav = np.sum(weights)

    def get_dict(self):
        dic = {}
        dic["center"] = self.center
        dic["num_samples"] = len(self.samples)
        dic["sample_ids"] = self.sample_ids
        dic["samples"] = self.samples
        dic["timestamps"] = self.timestamps
        dic["radius"] = self.radius
        dic["dm"] = self.dm
        dic["wav"] = self.wav
        dic["dist"] = self.dist
        dic["eps"] = self.eps
        dic["m"] = self.m
        return dic





import matplotlib.pyplot as plt
import matplotlib.patches as ptc
if __name__ == "__main__":
    make_fig = True
    stream, labels = load_data("complex9")  # [[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    gbstream = GBFuzzyStream(l=1)
    for i in range(len(stream)):
        cell_id, event_log = gbstream.learn_one(stream[i], ts[i])
        if (i + 1) % 1000 == 0:
            batch_start = max(0, i + 1 - 1000)
            print(i)
            summary = gbstream.get_summary()
            plt.figure(figsize=(8, 8))
            for fmb in summary.values():
                mc_patch = ptc.Circle((float(fmb.center[0]), float(fmb.center[1])), fmb.radius, alpha=0.4,
                                      color="lightgrey")
                plt.gca().add_patch(mc_patch)
                plt.scatter(x=fmb.center[0], y=fmb.center[1], alpha=1, c="black", s=50)
                plt.scatter(x=fmb.samples[:, 0], y=fmb.samples[:, 1], c="pink", alpha=1, marker="x", s=70)
            plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
                        cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
            plt.ylim((0, 1))
            plt.xlim((0, 1))
            plt.title("Number of FMBs: " + str(len(summary)) + " Number of Samples: " + str(
                sum([len(fmb.samples) for fmb in summary.values()])))
            plt.savefig(f"test_figures/gbfuzzystream/complex9_gbfuzzystream_{i}.png", bbox_inches='tight')
            plt.close()
            for j in np.arange(batch_start, i + 1):
                data_point = stream[j]
                problem = False
                for fmb in summary.values():
                    if j in fmb.sample_ids:
                        if data_point in fmb.samples:
                            print(j, "found")
                        else:
                            problem = fmb
                    if data_point in fmb.samples:
                        if j in fmb.sample_ids:
                            print(j, "rev. found")
                        else:
                            problem = fmb
                if problem:
                    print(j, data_point)
                    for fmb in summary.values():
                        if j in fmb.sample_ids:
                            print(fmb)
                        if data_point in fmb.samples:
                            print(fmb)

    # samples = np.array([[0,1], [3,5], [-1,4]])
    # fmb = FuzzyMicroBall(samples, np.array([0,1,2]), "euclidean", 0.5, 2)
    #
    # fmb1, fmb2, dm = fmb.split()
    # print(fmb1)
    # print(fmb2)
    # print(fmb.dm, dm)
    #print(fmb.iterate_child_centers(o1,o2))

