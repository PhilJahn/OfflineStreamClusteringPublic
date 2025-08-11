import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition


def get_offline_configspace(method, data_points, labels):
    config_space = ConfigurationSpace()

    d = len(data_points[0])
    gt_cls_num = len(np.unique(labels)) # use ground truth of batch

    if method == "dbscan":
        dbscan_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        dbscan_2 = Integer("min_samples", (1, 25), default=5)
        config_space.add([dbscan_1, dbscan_2])
    else:
        raise NotImplementedError(method)
    return config_space