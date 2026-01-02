import copy
import math

import numpy as np

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer


class SWClustering(StreamClusterer):

    def __init__(self, max_ehcf_num=100, epsilon=0.1, N=10000, beta=2.0, distance="euclidean", window=None):
        super().__init__()
        self.max_ehcf_num = max_ehcf_num
        self.epsilon = epsilon
        self.N = N
        self.beta = beta
        self.dist = distance

        self.ehcfs = {}
        self.ehcf_nn = {}

        self.ehcf_id = 0

        self.d_ctr = 0
        self.window = window

    def learn_one(self, x, t: int = None):
        if self.window is not None:
            self.t = self.d_ctr // self.window
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        x = np.array(x)
        event_log = ""
        x_tcf = make_tcf(x, t)
        closest_id, closest_distance = self.find_closest_ehcf_id(x, return_distance=True)
        inserted = False
        if closest_id is not None:
            closest_ehcf = self.ehcfs[closest_id]
            closest_boundary = closest_ehcf.boundary
            str_rep = str(closest_ehcf)
            print(str_rep)
            if "\\" in str_rep:
                raise Exception()
            closest_ehcf.calc_boundary()
            print(closest_boundary, closest_ehcf.boundary)
            assert closest_ehcf.boundary == closest_boundary
            if closest_distance <= closest_boundary:
                closest_ehcf.update(x_tcf)
                self.update_ehcf(closest_id)
                event_log += "insert '" + str(closest_id) + "';"
                inserted = True
                new_ehcf_id = closest_id
        if len(self.ehcfs.keys()) == self.max_ehcf_num:
            min_dist = np.inf
            min_ehcf_id = None
            for ehcf in self.ehcfs.values():
                ehcf_dist = ehcf.nn_dist
                if ehcf_dist < min_dist:
                    min_dist = ehcf_dist
                    min_ehcf_id = ehcf.id

            min_ehcf_id_2 = self.ehcf_nn[min_ehcf_id]
            ehcf1 = self.ehcfs[min_ehcf_id]
            ehcf2 = self.ehcfs[min_ehcf_id_2]
            #print(min_ehcf_id)
            #print(min_ehcf_id_2)
            #print(ehcf1.get_distance(ehcf2.mean))
            #print(min_dist)
            # print(ehcf1)
            # print(ehcf2)
            #print(self.find_closest_ehcf_id(ehcf1.mean, exclusion=[min_ehcf_id], return_distance=True))
            #print(self.find_closest_ehcf_id(ehcf2.mean, exclusion=[min_ehcf_id_2], return_distance=True))
            assert ehcf1.get_distance(ehcf2.mean) == min_dist



            self.delete_ehcf(min_ehcf_id)
            self.delete_ehcf(min_ehcf_id_2)

            new_cf1x = ehcf1.cf1x + ehcf2.cf1x
            new_cf2x = ehcf1.cf2x + ehcf2.cf2x
            new_t = max(ehcf1.t, ehcf2.t)
            new_n = ehcf1.n + ehcf2.n
            new_tcfs = {0:[]}
            new_ehcf_id = self.ehcf_id
            child_ehcfs = [ehcf1]


            new_ehcf = EHCF(ehcf_id=new_ehcf_id, cf1x=new_cf1x, cf2x=new_cf2x, n=new_n, tcfs=new_tcfs, t=new_t,
                            epsilon=self.epsilon, N=self.N, beta=self.beta, dist=self.dist, child_ehcfs=child_ehcfs)

            self.ehcfs[new_ehcf_id] = new_ehcf
            new_ehcf.merge_directly(ehcf2)
            new_ehcf.calc_boundary()
            self.update_ehcf(new_ehcf_id)
            self.ehcf_id = max(self.ehcfs.keys()) + 1

            event_log += "merge '" + str(min_ehcf_id) + "' and '"+ str(min_ehcf_id_2) + "' to '" + str(new_ehcf_id) + "';"

        assert len(self.ehcfs.keys()) <= self.max_ehcf_num-1
        if not inserted:
            new_ehcf_id = self.ehcf_id
            self.init_ehcf(x_tcf, ehcf_id=new_ehcf_id)
            self.update_ehcf(new_ehcf_id)
            event_log += "new '" + str(new_ehcf_id) + "';"

        del_ids = []
        for ehcf_id, ehcf in self.ehcfs.items():
            ehcf.age_out()
            if ehcf.n == 0: # empty ehcf
                del_ids.append(ehcf_id)

        for del_id in del_ids:
            self.delete_ehcf(del_id)

        return new_ehcf_id, event_log

    def update_ehcf(self, updated_ehcf_id):
        updated_ehcf = self.ehcfs[updated_ehcf_id]
        self.update_ehcf_nn(updated_ehcf_id)
        #print(self.ehcf_nn)
        for ehcf_id, closest_ehcf_id in self.ehcf_nn.items():
            if ehcf_id != updated_ehcf_id:
                ehcf = self.ehcfs[ehcf_id]
                dist = ehcf.get_distance(updated_ehcf.mean)
                if closest_ehcf_id == updated_ehcf_id:
                    #print(ehcf_id, "closest to ", updated_ehcf_id)
                    if dist > ehcf.nn_dist: #farther apart, need to find new closest
                        self.update_ehcf_nn(ehcf_id)
                    else:
                        ehcf.update_nn_dist(dist)
                else:
                    if dist < ehcf.nn_dist or (dist == ehcf.nn_dist and updated_ehcf_id < self.ehcf_nn[ehcf_id]): #closer than prior closest
                        self.ehcf_nn[ehcf_id] = updated_ehcf_id
                        ehcf.update_nn_dist(dist)

    def delete_ehcf(self, del_id):
        self.ehcfs.pop(del_id)
        if del_id in self.ehcf_nn.keys():
            self.ehcf_nn.pop(del_id)
        if del_id in self.ehcf_nn.values():
            for ehcf_id, closest_ehcf_id in self.ehcf_nn.items():
                if closest_ehcf_id == del_id:
                    self.update_ehcf_nn(ehcf_id)


    def learn(self, xs: list, timestamps: list = None):
        ehcf_ids = []
        event_logs = []
        for i in range(len(xs)):
            ehcf_id, event_log = self.learn_one(xs[i], timestamps[i])
            ehcf_ids.append(ehcf_id)
            event_logs.append(event_log)
        return ehcf_ids, event_logs


    def init_ehcf(self, initial_tcf, ehcf_id=None):
        if ehcf_id is None:
            ehcf_id = self.ehcf_id
        assert ehcf_id not in self.ehcfs.keys()
        tcfs = {0: [initial_tcf]}
        cf1x = copy.deepcopy(initial_tcf.cf1x)
        cf2x = copy.deepcopy(initial_tcf.cf2x)
        n = initial_tcf.n
        t = initial_tcf.t

        new_ehcf = EHCF(ehcf_id = ehcf_id, cf1x=cf1x, cf2x=cf2x, n=n, tcfs=tcfs, t=t, epsilon=self.epsilon, N=self.N,
                        beta=self.beta, dist=self.dist)

        self.ehcfs[ehcf_id] = new_ehcf



        self.ehcf_id = max(self.ehcfs.keys()) + 1


    def update_ehcf_nn(self, ehcf_id):
        ehcf = self.ehcfs[ehcf_id]
        nn, dist = self.find_closest_ehcf_id(ehcf.mean, [ehcf_id], return_distance=True)
        self.ehcf_nn[ehcf_id] = nn
        ehcf.update_nn_dist(dist)

    def find_closest_ehcf_id(self, pos, exclusion=None, return_distance=False):
        min_dist = np.inf
        closest_ehcf_id = None
        if exclusion is None:
            exclusion = []
        for ehcf_id, ehcf in self.ehcfs.items():
            if ehcf_id not in exclusion:
                dist = ehcf.get_distance(pos)
                if dist < min_dist:
                    closest_ehcf_id = ehcf_id
                    min_dist = dist
        if return_distance:
            return closest_ehcf_id, min_dist
        else:
            return closest_ehcf_id

    def get_summary(self):
        return copy.deepcopy(self.ehcfs)


class EHCF:
    def __init__(self, ehcf_id, cf1x, cf2x, n, tcfs, t, epsilon, N, beta, dist, child_ehcfs=None):
        self.id = ehcf_id
        self.tcfs = tcfs
        self.t = t
        self.cf1x = cf1x
        self.cf2x = cf2x
        self.n = n
        if child_ehcfs is None:
            self.child_ehcfs = []
        else:
            self.child_ehcfs = child_ehcfs
        self.mean = cf1x / n


        self.N = N
        self.max_per_level = math.ceil(1.0/epsilon) + 1
        self.beta = beta
        self.dist = dist

        self.nn_dist = None
        self.radius = None
        self.boundary = None
        self.calc_boundary()

        self.void_child = 0
        self.non_void_tcfs = len(self.get_all_tcfs())

    def get_distance(self, pos):
        return utils.distance_calc(self.mean, pos, self.dist)

    def calc_boundary(self):
        if self.n == 1:
            if self.nn_dist is None:
                self.radius = np.inf
            else:
                self.radius = self.nn_dist
        else:
            max_dist = 0
            ehcf_mean = self.mean
            for tcf in self.get_all_tcfs():
                tcf_mean = tcf.mean
                dist_tcf = utils.distance_calc(tcf_mean,  ehcf_mean, self.dist)
                if dist_tcf > max_dist:
                    max_dist = dist_tcf
            self.radius = max_dist
        self.boundary = self.radius * self.beta


    def update_nn_dist(self, nn_dist):
        self.nn_dist = nn_dist
        if self.n == 1:
            self.calc_boundary()

    def update(self, cur_tcf):
        i = 0
        self.t = max(cur_tcf.t, self.t)
        self.cf1x += cur_tcf.cf1x
        self.cf2x += cur_tcf.cf2x
        self.n += cur_tcf.n
        self.tcfs[i].append(cur_tcf)

        self.sweep_tcfs()

        if len(self.child_ehcfs) > 0:
            void_tcf = TCF(np.zeros(self.cf1x.shape), np.zeros(self.cf1x.shape), 0, 0, level=0, void=True)
            void_ehcf = self.child_ehcfs[self.void_child]
            start_void_child = self.void_child
            looped = False
            while len(void_ehcf.child_ehcfs) == 0 and void_ehcf.non_void_tcfs <= 1 and not looped:
                print(self.void_child)
                self.void_child += 1  # skip turn if already fully processed
                self.void_child %= len(self.child_ehcfs)
                looped = start_void_child == self.void_child
                void_ehcf = self.child_ehcfs[self.void_child]

            if not looped: # only looped when finished
                void_ehcf.update(void_tcf)
                void_ehcf.non_void_tcfs = len(void_ehcf.get_all_tcfs())

            for child_ehcf in self.child_ehcfs:
                if len(child_ehcf.child_ehcfs) > 0 or child_ehcf.non_void_tcfs > 1:
                    assert not looped

            merged = self.merge()
            if not merged:
                self.void_child += 1  # take turns voiding
                self.void_child %= len(self.child_ehcfs)

        self.age_out()

        sum_cf1x, sum_cf2x, sum_n = self.get_summed_values()
        #print(sum_cf1x, sum_cf2x, sum_n)
        #print(self.cf1x, self.cf2x, self.n)
        for i in range(len(self.cf1x)):
            assert sum_cf1x[i] - 0.001 <= self.cf1x[i] <= sum_cf1x[i] + 0.001
        self.mean = self.cf1x / self.n
        self.calc_boundary()

    def sweep_tcfs(self):
        i = 0
        while len(self.tcfs[i]) > self.max_per_level:
            #merge
            cur_tcf = self.tcfs[i].pop(0)
            cur_tcf.insert(self.tcfs[i].pop(0))
            cur_tcf.level = cur_tcf.level + 1
            if i+1 not in self.tcfs.keys():
                self.tcfs[i+1] = []
            self.tcfs[i+1].append(cur_tcf)
            if len(self.tcfs[i]) <= self.max_per_level:
                i += 1
        self.non_void_tcfs = len(self.get_all_tcfs())
        print(self)
        prior_t = np.inf
        for level in self.tcfs.keys():
            lvl_size = len(self.tcfs[level])
            assert lvl_size <= self.max_per_level
            for j in range(len(self.tcfs[level])):
                tcf = self.tcfs[level][lvl_size-j-1]
                if not tcf.void:
                    next_t = tcf.t
                    print(prior_t, tcf, next_t)
                    assert prior_t > next_t
                    prior_t = next_t

    def swap(self, other):
        temp_id = self.id
        temp_tcfs = self.tcfs
        temp_t = self.t
        temp_cf1x = self.cf1x
        temp_cf2x = self.cf2x
        temp_n = self.n
        temp_child_ehcfs = self.child_ehcfs
        temp_mean = self.mean
        temp_void_child = self.void_child
        temp_nn_dist = self.nn_dist
        temp_radius = self.radius
        temp_boundary = self.boundary
        temp_void_child = self.void_child
        temp_non_void_tcfs = self.non_void_tcfs

        self.id = other.id
        self.tcfs = other.tcfs
        self.t = other.t
        self.cf1x = other.cf1x
        self.cf2x = other.cf2x
        self.n = other.n
        self.child_ehcfs = other.child_ehcfs
        self.mean = other.mean
        self.void_child = other.void_child
        self.nn_dist = other.nn_dist
        self.radius = other.radius
        self.boundary = other.boundary
        self.void_child = other.void_child
        self.non_void_tcfs = other.non_void_tcfs

        other.id = temp_id
        other.tcfs = temp_tcfs
        other.t = temp_t
        other.cf1x = temp_cf1x
        other.cf2x = temp_cf2x
        other.n = temp_n
        other.child_ehcfs = temp_child_ehcfs
        other.mean = temp_mean
        other.void_child = temp_void_child
        other.nn_dist = temp_nn_dist
        other.radius = temp_radius
        other.boundary = temp_boundary
        other.void_child = temp_void_child
        other.non_void_tcfs = temp_non_void_tcfs



    def merge_directly(self, new_ehcf):
        new_min, new_max = new_ehcf.get_time_window_all()
        overlaps_self, older_self = self.check_overlap_self(new_min, new_max)
        print("'"+ str(self.id) + "'", self.get_time_window_self(), "vs", "'"+str(new_ehcf.id)+"'", new_min, new_max, "=", overlaps_self, older_self)
        if len(self.child_ehcfs) == 0: # bottom level
            if overlaps_self: # can't insert
                print("overlaps", "'"+ str(self.id) + "'")
                return False
            else: # insert
                ordered_new_tcfs = new_ehcf.get_ordered_own_tcfs()
                if not older_self: # all are newer, insert in reverse order in
                    if len(new_ehcf.child_ehcfs) == 0:
                        ordered_new_tcfs.reverse()
                        self.tcfs[0].extend(ordered_new_tcfs)
                        for tcf in ordered_new_tcfs:
                            tcf.level = 0
                        self.non_void_tcfs = len(self.get_all_tcfs())
                        self.t = new_ehcf.t
                        print("inserted front", "'"+ str(self.id) + "'")
                    else: # swap around
                        old_mean = new_ehcf.mean
                        self.swap(new_ehcf)
                        for d in range(len(old_mean)):
                            assert new_ehcf.mean[d] != old_mean[d] and self.mean[d] == old_mean[d]
                        print("swapped", "'"+ str(self.id) + "' and '"+ str(new_ehcf.id) + "'")
                        return self.merge_directly(new_ehcf) # insert original tcf into new one
                else:
                    i = max(self.tcfs.keys())
                    for tcf in ordered_new_tcfs:
                        self.tcfs[i].insert(0, tcf)
                        tcf.level = i
                    self.child_ehcfs = new_ehcf.child_ehcfs  # guaranteed to be older
                    self.non_void_tcfs = len(self.get_all_tcfs())
                    print("inserted back", "'"+ str(self.id) + "'")
                self.sweep_tcfs() # ensure
                return True
        elif len(self.child_ehcfs) == 1: # top-level
            if overlaps_self:
                print("overlaps", "'"+ str(self.id) + "'")
                self.child_ehcfs.append(new_ehcf)
                self.non_void_tcfs = len(self.get_all_tcfs())
                return None
            elif older_self: # not overlapping with root, older than root
                merged = self.child_ehcfs[0].merge_directly(new_ehcf)
                if merged: # inserted into some child_ehcf, since only one child, child is merged into self
                    print("collapse top", "'"+ str(self.id) + "'")
                    print("before")
                    print(self)
                    i = max(self.tcfs.keys())
                    ordered_new_tcfs = self.child_ehcfs[0].get_ordered_own_tcfs()
                    for j in range(len(ordered_new_tcfs)):
                        print(j, ordered_new_tcfs[j])
                    for j in range(len(ordered_new_tcfs)):
                        tcf = ordered_new_tcfs[-j-1]
                        self.tcfs[i].append(tcf)
                        print(tcf)
                        tcf.level = i
                    self.child_ehcfs = self.child_ehcfs[0].child_ehcfs
                    print("between")
                    print(self)
                    self.sweep_tcfs()
                    self.non_void_tcfs = len(self.get_all_tcfs())
                    print("after")
                    print(self)
                    return None
                else: # could not be inserted into any child_ehcf
                    print("could not insert", "'"+ str(self.id) + "'")
                    self.child_ehcfs.append(new_ehcf)
                    self.non_void_tcfs = len(self.get_all_tcfs())
                    return None
            else: # not overlapping with root, younger than root, root is just made -> not possible
                raise Exception

        else: # inner-node
            if overlaps_self:
                return False
            elif older_self: # not overlapping with self, older -> try to merge with bottom-most ehcf
                merged = False
                for i in range(len(self.child_ehcfs)):
                    if not merged:
                        merged = self.child_ehcfs[i].merge_directly(new_ehcf)
                if merged:
                    self.non_void_tcfs = len(self.get_all_tcfs())
                return merged
            else: # not overlapping with self, but newer -> try to merge with bottom-most ehcf of new_ehcf
                old_mean = new_ehcf.mean
                self.swap(new_ehcf)
                for d in range(len(old_mean)):
                    assert new_ehcf.mean[d] != old_mean[d] and self.mean[d] == old_mean[d]
                print("swapped in tree")
                return self.merge_directly(new_ehcf)


    def merge(self):
        # check if can be merged
        print("merge check", "'"+ str(self.id) + "'")
        assert len(self.child_ehcfs) > 1
        can_merge = True
        last_children = []
        for child_ehcf in self.child_ehcfs:
            if len(child_ehcf.child_ehcfs) > 0 or child_ehcf.non_void_tcfs > 1:
                can_merge = False
            else:
                print(child_ehcf)
                print(child_ehcf.non_void_tcfs, child_ehcf.get_all_tcfs())
                last_children.extend(child_ehcf.get_all_tcfs())
        if can_merge:
            print("can merge", "'"+ str(self.child_ehcfs[0].id) + "' and '"+ str(self.child_ehcfs[1].id) + "'")
            print("pre-merge", self)
            assert len(last_children) <= 2
            max_level = max(self.tcfs.keys())
            insertion_level = max_level
            if len(self.tcfs[insertion_level]) == self.max_per_level:
                insertion_level += 1
                self.tcfs[insertion_level] = []
            if len(last_children) == 2:
                print(last_children)
                child_tcf = last_children[0]
                child_tcf.insert(last_children[1])
            else:
                child_tcf = last_children[0]
            self.tcfs[insertion_level].insert(0, child_tcf)
            self.child_ehcfs = []
            self.void_child = 0
            print("post-merge")
            print(self)
            self.sweep_tcfs()
            return True
        else:
            return False

    def get_time_window_self(self):
        if len(self.tcfs[0]) > 0:
            own_max = self.t
            own_min = self.tcfs[max(self.tcfs.keys())][0].t
            return own_min, own_max
        else:
            return np.inf, 0

    def check_overlap_self(self, min_t, max_t):
        own_min, own_max = self.get_time_window_self()
        if max_t < own_min:
            return False, True # no overlap, is older
        elif min_t > own_max:
            return False, False # no overlap, is younger
        else:
            return True, True # overlap, order irrelevant


    def get_time_window_all(self):
        all_tcfs = self.get_all_tcfs()
        own_min = np.inf
        own_max = 0
        for tcf in all_tcfs:
            own_min = min(own_min, tcf.t)
            own_max = max(own_max, tcf.t)
        return own_min, own_max

    def check_overlap_all(self, min_t, max_t):
        own_min, own_max = self.get_time_window_all()
        if max_t < own_min:
            return False, True # no overlap, is older
        elif min_t > own_max:
            return False, False # no overlap, is younger
        else:
            return True, True # overlap, order irrelevant




    def age_out(self):
        highest_level = max(self.tcfs.keys())
        #print(self)
        highest_level_tcfs = self.tcfs[highest_level]
        if len(highest_level_tcfs) > 0:
            oldest = highest_level_tcfs[0]
            if oldest.t < self.t - self.N:
                self.tcfs[highest_level].pop(0)
                self.cf1x -= oldest.cf1x
                self.cf2x -= oldest.cf2x
                self.n -= oldest.n
                if len(self.tcfs[highest_level]) == 0 and highest_level > 0:
                    self.tcfs.pop(highest_level)
                self.calc_boundary()
                self.non_void_tcfs -= 1

        del_children = []
        for child_ehcf in self.child_ehcfs:
            child_ehcf.age_out()
            if child_ehcf.n == 0:
                del_children.append(child_ehcf)
        for del_child in del_children:
            self.child_ehcfs.remove(del_child)




    def get_summed_values(self):
        sum_cf1x = np.zeros(len(self.cf1x))
        sum_cf2x = np.zeros(len(self.cf2x))
        sum_n = 0
        for level in self.tcfs.values():
            for tcf in level:
                sum_cf1x += tcf.cf1x
                sum_cf2x += tcf.cf2x
                sum_n += tcf.n
        for child_ehcf in self.child_ehcfs:
            ccf1x, ccf2x, cn = child_ehcf.get_summed_values()
            sum_cf1x += ccf1x
            sum_cf2x += ccf2x
            sum_n += cn
        return sum_cf1x, sum_cf2x, sum_n

    def get_dict(self):
        dic = {}
        dic["id"] = str(self.id)
        #dic["cf1x"] = self.cf1x
        #dic["cf2x"] = self.cf2x
        dic["n"] = self.n
        dic["t"] = self.t
        #dic["mean"] = self.mean
        tcfs = {}
        for key, level in self.tcfs.items():
            tcfs[key] = [str(tcf) for tcf in level]
        dic["tcfs"] = tcfs
        children = []
        for child_ehcf in self.child_ehcfs:
            children.append(str(child_ehcf))
        dic["children"] = children
        return dic

    def get_all_tcfs(self):
        tcfs = []
        for level in self.tcfs.values():
            for tcf in level:
                if not tcf.void:
                    tcfs.append(tcf)
        for child_ehcf in self.child_ehcfs:
            tcfs.extend(child_ehcf.get_all_tcfs())
        return tcfs

    def get_ordered_own_tcfs(self):
        tcfs = []

        for level in self.tcfs.keys():
            lvl_size = len(self.tcfs[level])
            for j in range(len(self.tcfs[level])):
                tcf = self.tcfs[level][lvl_size-j-1]
                if not tcf.void:
                    tcfs.append(tcf)
        return tcfs

    def __str__(self):
        return str(self.get_dict()).replace("\\", "")

class TCF:
    def __init__(self, cf1x, cf2x, t, n, level, void=False):
        self.cf1x = cf1x
        self.cf2x = cf2x
        self.t = t
        self.n = n
        self.mean = cf1x / n
        self.level = level
        self.void = void

    def insert_instance(self, instance, t):
        instance_tcf = make_tcf(instance, t)
        self.insert(instance_tcf)

    def insert(self, tcf):
        self.t = max(self.t, tcf.t)
        self.n = self.n + tcf.n
        self.cf1x = self.cf1x + tcf.cf1x
        self.cf2x = self.cf2x + tcf.cf2x
        self.mean = self.cf1x/self.n

        if self.void:
            self.void = tcf.void # in-case merge would turn non-void into void

    def get_dict(self):
        dic = {}
        #dic["cf1x"] = self.cf1x
        #dic["cf2x"] = self.cf2x
        dic["n"] = self.n
        dic["t"] = self.t
        #dic["mean"] = self.mean
        dic["level"] = self.level
        #dic["void"] = self.void
        return dic

    def __str__(self):
        return str(self.get_dict()).replace("\\", "")

def make_tcf(instance, t):
    cf2x = []
    cf1x = []
    for d in range(len(instance)):
        cf2x.append(float(instance[d]) ** 2)
        cf1x.append(float(instance[d]))
    return TCF(np.array(cf1x), np.array(cf2x), t, 1, level=0)


import matplotlib.pyplot as plt
import matplotlib.patches as ptc
if __name__ == "__main__":

    stream, labels = load_data("densired2")  #[[0],[1],[3],[7],[8],[9]]
    #stream = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[12],[13],[14],[15],[16], [17], [2.5], [2.6]]
    ts = list(range(len(stream)))
    make_fig = False

    #{'N': 2000, 'batch_budget': 1800, 'batch_size': 1000, 'beta': 2.1208140355588, 'ds': 'densired2', 'epsilon': 0.7756068704238, 'mmc': 100, 'offline': None, 'online': 'swclustering'}


    swclustering = SWClustering(epsilon=0.7756068704238, N=2000, beta=2.1208140355588, max_ehcf_num=100)
    for i in range(len(stream)):
        ehcf_id, event_log = swclustering.learn_one(stream[i], ts[i])
        print("TS", i, event_log)
        if make_fig:
            ehcfs = swclustering.get_summary()
            merge_id = None
            if "merge" in event_log:
                event_log_split = event_log.split(";")
                for event in event_log_split:
                    if "merge" in event:
                        merge_id = event.split(" ")[1]

            plt.figure(figsize=(8, 8))
            batch_start = max(0, i + 1 - 500)
            for ehcf_id, ehcf in ehcfs.items():
                # print(ehcf.mean)
                all_tcfs = ehcf.get_all_tcfs()
                ehcf_patch = ptc.Circle((float(ehcf.mean[0]), float(ehcf.mean[1])), ehcf.radius, alpha=0.4, color="lightgrey")
                plt.scatter(x=ehcf.mean[0], y=ehcf.mean[1], c=ehcf_id, alpha=1, s=100)
                for tcf in all_tcfs:
                    plt.scatter(x=tcf.mean[0], y=tcf.mean[1], c=ehcf_id, marker='*', alpha=0.5, s=30)
                    #https://stackoverflow.com/questions/20130227/how-to-connect-scatterplot-points-with-line-using-matplotlib
                    plt.plot([ehcf.mean[0], tcf.mean[0]], [ehcf.mean[1], tcf.mean[1]], '-o', c="grey")
                    plt.text(tcf.mean[0] + 0.02, tcf.mean[1], s=tcf.level)

                plt.gca().add_patch(ehcf_patch)
                if merge_id is not None:
                    if int(merge_id) == int(ehcf_id):
                        print("merged")
                        plt.scatter(x=ehcf.mean[0], y=ehcf.mean[1], c="pink", alpha=1, marker="x", s=150)
            plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
                        cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
            plt.scatter(x=np.array(stream)[i:i + 1][:, 0], y=np.array(stream)[i:i + 1][:, 1], c="pink", s=25)
            plt.ylim(-0.02,1.02)
            plt.xlim(-0.02, 1.02)
            plt.savefig(f"test_figures/swclustering/complex9_swclustering_{i}.png", bbox_inches='tight')
            plt.close()
