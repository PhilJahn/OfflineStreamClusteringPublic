from pprint import pprint

import numpy as np
import itertools
import ast
import copy


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