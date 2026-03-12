# Original code from xjnine and 920850946 (dmg) , from https://github.com/xjnine/GBFuzzyStream/tree/master

# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:35:51 2022

@author: xiejiang
"""

import numpy as np
import matplotlib.pyplot as plt
from original_implementations.gbfuzzystream import FuzzyClustering_no_random

verbose =False

class GB:
    def __init__(self, data, label):
        self.data = data
        self.center = self.data[:, :-1].mean(0)
        self.radius = self.get_radius()
        self.label = label
        self.num = len(data)

    def get_radius(self):
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)


def get_density_volume(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        if verbose:
            print("0")
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num
    return density_volume


# No parameter traversal to see if the ball needs to be split.
# Split without breaking judgment based on the comparison between the child ball and the parent ball, 1 point 2
def division_2_2(gb_list, m, eps):
    gb_list_new = []
    #print([len(gb_data) for gb_data in gb_list])
    for gb_data in gb_list:
        if len(gb_data) >= 1:
            ball_1, ball_2 = spilt_ball_fuzzy(gb_data, m, eps)  # 有模糊
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                continue
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                continue
            parent_dm = get_density_volume(gb_data[:, :-1])
            child_1_dm = get_density_volume(ball_1[:, :-1])
            child_2_dm = get_density_volume(ball_2[:, :-1])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))
            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                #print(w_child_dm, "<=", parent_dm)
                gb_list_new.append(gb_data)
        else:
            #print("more than 8")
            gb_list_new.append(gb_data)
    #print("split Hyperballs", len(gb_list_new))
    return gb_list_new


# fuzzy ball splitting
def spilt_ball_fuzzy(data, m, eps):
    cluster = FuzzyClustering_no_random.FCM_no_random(data[:, :-1], 2, m, eps)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def get_radius(gb_data):
    # origin get_radius 7*O(n)
    sample_num = len(gb_data)
    #print(gb_data[:,:-1])
    center = gb_data[:,:-1].mean(0)
    #print(center)
    diffMat = np.tile(center, (sample_num, 1)) - gb_data[:,:-1]
    #print(diffMat)
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius


def gb_plot(gb_dict, noise, t):
    cluster_label_list = []
    for i in range(0, len(gb_dict)):
        if gb_dict[i].label not in cluster_label_list:
            cluster_label_list.append(gb_dict[i].label)
    #plt.figure(figsize=(15, 15))
    for i in range(0, len(cluster_label_list)):
        if cluster_label_list[i] == -1:
            cluster_label_list.remove(-1)
            break
    cluster = {}
    for label in cluster_label_list:
        for key in gb_dict.keys():
            if gb_dict[key].label == label:
                if label not in cluster.keys():
                    cluster[label] = gb_dict[key].data
                else:
                    cluster[label] = np.append(cluster[label], gb_dict[key].data, axis=0)
    theta = np.arange(0, 2 * np.pi, 0.01)
    for i, key in enumerate(cluster.keys()):
        #plt.scatter(cluster[key][:, 0], cluster[key][:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
        #            marker='o', label=label_c[i])
        # 画圆
        for gb in gb_dict.values():
            if gb.label == key:
                center = gb.data[:, :-1].mean(0)
                r = gb.get_radius()
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                #plt.plot(x, y, c=color[i])
    #if len(noise) > 0:
    #    plt.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')
    #for key in gb_dict.keys():
    #    for i in range(0, len(cluster_label_list)):
    #        if gb_dict[key].label == -1:
     #           plt.scatter(gb_dict[key].data[:, 0], gb_dict[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
    #                        marker='x')
    #plt.legend(loc=1, fontsize=12)
    #plt.title("no." + str(t), size=20, loc='center')
    #plt.show()


# Shrink the ball
def minimum_ball(gb_list, radius_detect, m, eps):
    gb_list_temp = []
    for gb_data in gb_list:
        if len(gb_data) <= 2:
            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.2 * radius_detect):
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
            else:
                gb_list_temp.append(gb_data)
        else:
            if get_radius(gb_data) <= 1.2 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                ball_1, ball_2 = spilt_ball_fuzzy(gb_data, m, eps)  # 有模糊
                if len(ball_1) == 1 or len(ball_2) == 1:
                    if get_radius(gb_data) > radius_detect:
                        gb_list_temp.extend([ball_1, ball_2])
                    else:
                        gb_list_temp.append(gb_data)
                else:
                    gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


# Normalization
def normalized_ball(gb_list, radius_detect, m, eps):
    gb_list_temp = []
    for gb_data in gb_list:
        if len(gb_data) <= 2:
            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.5 * radius_detect):
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
            else:
                gb_list_temp.append(gb_data)
        else:
            if get_radius(gb_data[:, :-1]) <= 1.5 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                ball_1, ball_2 = spilt_ball_fuzzy(gb_data, m, eps)  # 有模糊
                gb_list_temp.extend([ball_1, ball_2])
    return gb_list_temp

