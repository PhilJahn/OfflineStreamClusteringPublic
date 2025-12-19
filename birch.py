import copy

import numpy as np

import utils
from datahandler import load_data
from online.stream_clustering import StreamClusterer


class Birch(StreamClusterer):

    def __init__(self, max_leaf_entry_num=1000, L=4, B=2, closeness="D0", absorption="R", outlier_ratio=0.1, outlier_batch=100):
        super().__init__()
        self.threshold = 0
        self.leaf_entry_ctr = 0
        self.L = L
        self.B = B
        self.closeness = closeness
        self.absorption = absorption
        self.max_leaf_entry_num = max_leaf_entry_num
        self.cf_tree_root = CFNode(parent=None, children=[], L=self.L, B=self.B, threshold=self.threshold,
                                   closeness=self.closeness, absorption=self.absorption)
        self.d_ctr = 0
        self.left_most_leaf = self.cf_tree_root
        self.outlier_ratio = outlier_ratio

        self.outlier_disk = []
        self.outlier_batch = outlier_batch

    def find_new_threshold_and_outlier_ratio(self) -> (float, float):
        closenesses = []
        size = []
        node = self.left_most_leaf
        while node is not None:
            for child1 in node.children:
                min_closesness = np.inf
                if len(node.children) > 1: # ignore if no other entry in leaf
                    for child2 in node.children:
                        if child1 != child2:
                            cand_closeness = compute_closeness_nodes(self.closeness, child1, child2)
                            if cand_closeness < min_closesness:
                                min_closesness = cand_closeness#
                    closenesses.append(min_closesness)
                size.append(child1.n)
            node = node.right
        #print(closenesses)
        return np.mean(closenesses), np.mean(size) # Betula prefers median, but doesn't use it

    def get_all_entries(self):
        entries = []
        node = self.left_most_leaf
        while node is not None:
            entries.extend(node.children)
            node = node.right
        return entries

    def learn_one(self, x, t: int = None):
        event_log = ""
        self.d_ctr += 1
        x = np.array(x)
        virtual_leaf_node = CFLeafEntry(n=1, mean=x, S=0, S_vec=np.zeros(len(x)))
        self.cf_tree_root.insert(virtual_leaf_node)
        if self.cf_tree_root.parent is not None:
            self.cf_tree_root = self.cf_tree_root.parent
        assert self.cf_tree_root.parent is None
        if self.cf_tree_root.entry_count >= self.max_leaf_entry_num:
            self.threshold, avg_entry_size = self.find_new_threshold_and_outlier_ratio()
            outlier_threshold = self.outlier_ratio * avg_entry_size
            #print("thresholds", self.threshold, outlier_threshold)
            new_root = CFNode(parent=None, children=[], L=self.L, B=self.B, threshold=self.threshold,
                                       closeness=self.closeness, absorption=self.absorption)
            self.left_most_leaf = new_root
            old_count = self.cf_tree_root.n
            node = self.cf_tree_root
            #print(self.cf_tree_root)
            while node is not None:
                if (not node.is_leaf) and len(node.children) > 0: # edge case of root is leaf-node
                    node = node.children[0]
                #print(str([node]).split(" ")[-1][:-2])
                if node.is_leaf:
                    for entry in node.children:
                        entry.update_parent(None)
                        #print("size", entry.n)
                        if entry.n >= outlier_threshold:
                            new_root.insert(entry)
                            if new_root.parent is not None:
                                new_root = new_root.parent
                                assert new_root.parent is None
                            #print(new_root)
                        else:
                            self.outlier_disk.append(entry)
                    node.children = []
                if len(node.children) == 0:
                    node_parent = node.parent
                    #print("empty", str([node]).split(" ")[-1][:-2])
                    if node_parent is not None:
                        node_parent.children.remove(node)
                        del node
                    node = node_parent
                    #print("move up", str([node]).split(" ")[-1][:-2])
                event_log += "reconstructed;"
            #print("old root", self.cf_tree_root)
            self.cf_tree_root = new_root
            #print("new root", self.cf_tree_root)
            outlier_n = sum([outlier.n for outlier in self.outlier_disk])
            #print(self.cf_tree_root.n)
            #print(outlier_n)
            #print(old_count)
            assert self.cf_tree_root.n + outlier_n == old_count
            reinserted = []
            for outlier in self.outlier_disk:
                _, inserted = self.cf_tree_root.insert(outlier, outlier=True)
                if inserted:
                    reinserted.append(outlier)
            for reinsert in reinserted:
                self.outlier_disk.remove(reinsert)
                event_log += "reins;"
        if self.d_ctr % self.outlier_batch == 0: # periodically check for reinsertion
            #print("Check outliers")
            reinserted = []
            for outlier in self.outlier_disk:
                _, inserted = self.cf_tree_root.insert(outlier, outlier=True)
                if inserted:
                    reinserted.append(outlier)
            for reinsert in reinserted:
                self.outlier_disk.remove(reinsert)

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

        summary_root = copy.deepcopy(self.cf_tree_root)
        summary_outlier_disk = copy.deepcopy(self.outlier_disk)
        reinserted = []

        if self.d_ctr % self.outlier_batch == 0:  # check for reinsertion at the end -> do it on summary call without
            for outlier in summary_outlier_disk:
                _, inserted = summary_root.insert(outlier, outlier=True)
                if inserted:
                    reinserted.append(outlier)
        for reinsert in reinserted:
            summary_outlier_disk.remove(reinsert)

        node = summary_root
        while len(node.children) > 0 and not node.is_leaf:
            node = node.children[0]
        while node.left is not None:
            node = node.left
        assert node.left is None
        summary_entries = []
        while node is not None:
            summary_entries.extend(copy.deepcopy(node.children))
            node = node.right
        return {"entries": summary_entries, "outliers": summary_outlier_disk, "tree": summary_root}


class CFNode():
    def __init__(self, parent, children, L, B, threshold, closeness, absorption, is_leaf=True, left=None, right=None):
        self.parent = parent
        self.children = children
        self.left = left
        self.right = right
        self.entry_count = 0

        self.is_leaf = is_leaf

        self.threshold = threshold
        self.L = L
        self.B = B
        self.closeness = closeness
        self.absorption = absorption

        self.n = 0
        self.mean = None
        self.S = None
        self.S_vec = None

        if len(self.children) > 0:
            self.propagate_compute_features()
            if self.is_leaf:
                assert type(self.children[0]) == CFLeafEntry
            else:
                assert type(self.children[0]) != CFLeafEntry

    def update_left(self, new_left):
        self.left = new_left
        assert self.is_leaf

    def update_right(self, new_right):
        self.right = new_right
        assert self.is_leaf

    def update_parent(self, new_parent):  # first make child, then make parent
        self.parent = new_parent
        assert self in self.parent.children

    def compute_features(self):
        self.n = self.children[0].n
        self.mean = self.children[0].mean
        self.S = self.children[0].S
        self.S_vec = self.children[0].S_vec
        self.entry_count = self.children[0].entry_count
        for i in range(1, len(self.children)):
            child = self.children[i]
            nA = self.n
            meanA = self.mean
            SA = self.S
            SA_vec = self.S_vec
            nB = child.n
            meanB = child.mean
            SB = child.S
            SB_vec = child.S_vec
            self.n, self.mean, self.S, self.S_vec = compute_features(nA, meanA, SA, SA_vec, nB, meanB, SB, SB_vec)
            self.entry_count += child.entry_count


    def insert(self, x, outlier=False):
        if len(self.children) == 0:  # at initialization
            assert self.parent is None
            if outlier:
                return False, False
            self.children = [x]
            x.update_parent(self)
        else:  # pick best suited child
            best_i = None
            best_closeness = np.inf
            for i in range(len(self.children)):
                child = self.children[i]
                close = compute_closeness_nodes(self.closeness, child, x)
                if close < best_closeness:
                    best_i = i
                    best_closeness = close

            if self.is_leaf:
                # check if merged or new entry
                #print("arrived at leaf")
                child = self.children[best_i]
                absorption_value = compute_absorption_nodes(self.absorption, child, x)
                if absorption_value <= self.threshold:
                    #print("Absorbed")
                    child.insert(x)
                else:
                    if outlier:
                        return False, False
                    #print("Not absorbed")
                    self.children.append(x)
                    x.update_parent(self)
                    if len(self.children) > self.L:  # splitting needed
                        #print("Split needed", len(self.children))
                        new_root = self.split()
                        return new_root
            else:
                #print("Move down")
                self.children[best_i].insert(x)

        if self.is_leaf:
            assert type(self.children[0]) == CFLeafEntry
        else:
            assert type(self.children[0]) != CFLeafEntry

        self.propagate_compute_features()
        if outlier:
            return False, True
        else:
            return False

    def propagate_compute_features(self):
        self.compute_features()
        if self.parent is not None:
            self.parent.propagate_compute_features()


    def propagate_split(self, new_child1, new_child2):
        assert not self.is_leaf
        self.children.append(new_child2)
        new_child2.update_parent(self)
        assert new_child1.parent is self
        if len(self.children) <= self.B * 2:
            #print("no split for parent")
            self.propagate_compute_features()
            pair1, pair2, _ = self.find_closest_children()
            if not ((pair1 == new_child1 and pair2 == new_child2) or (pair2 == new_child1 and pair1 == new_child2)):
                if pair2.left is not None: # don't remove first leaf entry
                    #print("pair 1", pair1)
                    #print("pair 2", pair2)
                    #print(pair1.left, pair2.left)
                    self.children.remove(pair2)
                    pair1.merge(pair2)
                else:
                    #print("pair 1", pair1)
                    #print("pair 2", pair2)
                    #print(pair1.left, pair2.left)
                    self.children.remove(pair1)
                    pair2.merge(pair1)
            return False
        else:
            #print("split for parent")
            return self.split()

    def find_closest_children(self):
        closest_i = None
        closest_j = None
        closest_closeness = np.inf
        for i in range(len(self.children)):
            for j in range(i + 1, len(self.children)):
                close = compute_closeness_nodes(self.closeness, self.children[i], self.children[j])
                if close < closest_closeness:
                    closest_i = i
                    closest_j = j
                    closest_closeness = close
        return self.children[closest_i], self.children[closest_j], closest_closeness

    def merge(self, x):
        #print("merge")
        self.children.extend(x.children)
        for child in x.children:
            child.update_parent(self)
        x_left = x.left
        x_right = x.right
        if self.is_leaf:
            x_left.update_right(x_right)
        if x_right is not None:
            x_right.update_left(x_left)
        self.propagate_compute_features()
        if self.is_leaf:
            if len(self.children) > self.L:
                return self.split()
            else:
                return False
        else:
            if len(self.children) > self.B * 2:
                return self.split()
            else:
                return False



    def split(self):
        farthest_i = None
        farthest_j = None
        farthest_closeness = 0
        for i in range(len(self.children)):
            for j in range(i + 1, len(self.children)):
                close = compute_closeness_nodes(self.closeness, self.children[i], self.children[j])
                if close > farthest_closeness:
                    farthest_i = i
                    farthest_j = j
                    farthest_closeness = close
        i_children = []
        j_children = []
        #print("split seeds", self.children[farthest_i], self.children[farthest_j])
        for h in range(len(self.children)):
            close_i = compute_closeness_nodes(self.closeness, self.children[h], self.children[farthest_i])
            close_j = compute_closeness_nodes(self.closeness, self.children[h], self.children[farthest_j])
            if close_i < close_j:
                i_children.append(self.children[h])
            else:
                j_children.append(self.children[h])
        #i_node = CFNode(self.parent, i_children, is_leaf = self.is_leaf, L=self.L, B=self.B, threshold=self.threshold, closeness=self.closeness, absorption = self.absorption)      
        # i_node is self, j_node is new
        #print("before", self)
        #print(self.children[farthest_i], ":", i_children[:], "; ", self.children[farthest_j], ":", j_children[:])
        self.children = i_children
        #print(self.entry_count, len(i_children), len(j_children))
        #print("after", self)

        j_node = CFNode(parent=self.parent, children=j_children, is_leaf=self.is_leaf, L=self.L, B=self.B, threshold=self.threshold,
                        closeness=self.closeness, absorption=self.absorption)
        assert j_node.parent == self.parent
        if self.is_leaf:
            #print("I was here")
            j_node.update_left(self)
            j_node.update_right(self.right)
            if self.right is not None:
                self.right.update_left(j_node)
            self.update_right(j_node)
        for child in j_children:
            child.update_parent(j_node)

        self.propagate_compute_features()

        if self.parent is not None:  # not root
            return self.parent.propagate_split(self, j_node)
        else:
            self.parent = CFNode(parent=None, children=[self, j_node], is_leaf=False, L=self.L, B=self.B,
                                 threshold=self.threshold, closeness=self.closeness, absorption=self.absorption)
            j_node.update_parent(self.parent)
            return self.parent

    def get_dict(self):
        dic = {}
        dic['self'] = str([self]).split(" ")[-1][:-2]
        dic['len_children'] = len(self.children)
        dic['parent'] = str([self.parent]).split(" ")[-1][:-2]
        dic['n'] = self.n
        dic['mean'] = self.mean
        dic['entry'] = self.entry_count
       # dic['S'] = self.S
       # dic['S_vec'] = self.S_vec
        dic['leaf'] = self.is_leaf
        dic['left'] = str([self.left]).split(" ")[-1][:-2]
        dic['right'] = str([self.right]).split(" ")[-1][:-2]
        #dic['absorption'] = self.absorption
        #dic['closeness'] = self.closeness
        #dic['threshold'] = self.threshold
        #dic['B'] = self.B
       # dic['L'] = self.L
        return dic

    def __str__(self):
        rep = "["
        for child in self.children:
            rep += str(child)
            rep += ", "
        rep = rep[:-2]
        rep += "] " + str(self.get_dict()) + "\n"
        return rep


class CFLeafEntry():
    def __init__(self, n, mean, S, S_vec):
        self.n = n
        self.entry = [mean]
        self.mean = mean
        self.S = S
        self.S_vec = S_vec
        self.parent = None
        self.radius = (self.S / self.n) ** 0.5
        self.entry_count = 1

    def update_parent(self, new_parent):
        self.parent = new_parent

    def insert(self, x):
        #print("Leaf insert")
        self.n, self.mean, self.S, self.S_vec = compute_features_nodes(self, x)
        self.radius = (self.S / self.n) ** 0.5
        self.entry.extend(x.entry)

    def get_dict(self):
        dic = {}
        dic['n'] = self.n
        #dic["entry"] = self.entry
        dic['mean'] = self.mean
        dic['parent'] = str([self.parent]).split(" ")[-1][:-2]
        dic['S'] = self.S
        #dic['S_vec'] = self.S_vec
        #dic['radius'] = self.radius
        return dic

    def __str__(self):
        return str(self.get_dict())


def compute_features_nodes(nodeA, nodeB):
    return compute_features(nodeA.n, nodeA.mean, nodeA.S, nodeA.S_vec, nodeB.n, nodeB.mean, nodeB.S, nodeB.S_vec)


def compute_features(nA, meanA, SA, SA_vec, nB, meanB, SB, SB_vec):
    nAB = nA + nB
    meanAB = meanA + (meanB - meanA) * (nB / nAB)
    SAB = SA + SB + (nB * np.dot((meanB - meanA), (meanB - meanAB)))
    SAB_vec = SA_vec + SB_vec + (nB * (meanB - meanA) * (meanB - meanAB))
    return nAB, meanAB, SAB, SAB_vec


def compute_closeness_nodes(closeness, nodeA, nodeB):
    return compute_closeness(closeness, nodeA.n, nodeA.mean, nodeA.S, nodeB.n, nodeB.mean, nodeB.S)


def compute_closeness(closeness, nA, meanA, SA, nB, meanB, SB):
    if closeness == "D0":
        return utils.distance_calc(meanA, meanB, "euclidean")
    elif closeness == "D1":
        return utils.distance_calc(meanA, meanB, "manhattan")
    elif closeness == "D2":
        euc_dist = utils.distance_calc(meanA, meanB, "euclidean")
        return (1 / nA * SA + 1 / nB * SB + (euc_dist ** 2)) ** 0.5
    elif closeness == "D3":
        euc_dist = utils.distance_calc(meanA, meanB, "euclidean")
        nAB = nA + nB
        inner = nAB * (SA + SB) + nA + nB * (euc_dist ** 2)
        return (2 * inner / (nAB * (nAB - 1))) ** 0.5
    elif closeness == "D4":
        euc_dist = utils.distance_calc(meanA, meanB, "euclidean")
        nAB = nA + nB
        return (nA * nB * (euc_dist ** 2) / nAB) ** 0.5
    else:
        raise ValueError("Invalid closeness parameter")


def compute_absorption_nodes(absorption, nodeA, nodeB):
    if absorption == "euclidean":
        return utils.distance_calc(nodeA.mean, nodeB.mean, "euclidean")
    else:
        nAB, meanAB, SAB, SAB_vec = compute_features_nodes(nodeA, nodeB)
        return compute_absorption(absorption, nAB, meanAB, SAB)


def compute_absorption(absorption, n, mean, S):
    if absorption == "R":
        return (S / n) ** 0.5
    elif absorption == "D":
        return (2 * S / (n - 1)) ** 0.5
    else:
        raise ValueError("Invalid absorption parameter")

import matplotlib.pyplot as plt
import matplotlib.patches as ptc
if __name__ == "__main__":
    stream, labels = load_data("complex9")  #[[0],[1],[3],[7],[8],[9]]
    #stream = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[12],[13],[14],[15],[16], [17], [2.5], [2.6]]
    ts = list(range(len(stream)))
    make_fig = True
    betula = Birch(max_leaf_entry_num=100, closeness="D0")
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


