from pprint import pprint

import numpy as np
import itertools
import ast
import copy

from sklearn.metrics import pairwise_distances


def dict_to_np(dp):
    value_list = [value for key, value in sorted(dp.items())]
    return np.array(value_list)


def dps_to_np(dps):
    dps_list = []
    for dp in dps:
        dp_list = dict_to_np(dp)
        dps_list.append(dp_list)
    return np.array(dps_list)

def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)

def get_noise_labels(labels):
    labels = copy.deepcopy(labels)
    min_label = -1
    for i in np.where(labels == -1)[0]:
        min_label -= 1
        labels[i] = min_label

    return labels

def get_data_properties(data_points, labels):
    d = len(data_points[0])
    all_lb = np.unique(labels)
    gt_cls = np.where(all_lb >= 0)[0]
    gt_cls_num = max(1,len(gt_cls))
    return d, gt_cls_num

def distance_calc(pos1, pos2, dist):
    if dist == "euclidean":
        if len(pos1.shape) == 1:
            return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))
        elif len(pos2.shape) == 1:
            print(pos1.shape, pos2.shape)
            print(pos2)
            return np.array([np.sqrt(np.sum(np.square(pos - pos2), axis=-1)) for pos in pos1])
        else:
            return np.array([[np.sqrt(np.sum(np.square(pos - post), axis=-1)) for post in pos2] for pos in pos1])
    elif dist == "manhattan":
        return np.sum(np.abs(pos1 - pos2), axis=-1)
    else:
        if len(pos1.shape) == 1 and len(pos2.shape) == 1:
            return pairwise_distances(pos1.reshape(1, -1), pos2.reshape(1, -1), dist)[0][0]
        elif len(pos1.shape) == 1:
            return pairwise_distances(pos1.reshape(1, -1), pos2, dist)[0]
        elif len(pos2.shape) == 1:
            return pairwise_distances(pos1, pos2.reshape(1, -1), dist).flatten()
        else:
            return pairwise_distances(pos1, pos2, dist)

if __name__ == "__main__":
    pos1 = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    pos2 = np.array([[3,4,5], [6,7,8], [9,10,11]])
    dist = distance_calc(pos1, pos2, "euclidean")
    print(dist, dist.shape)