import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition


def get_online_configspace(method, batch, data_points, labels):
    config_space = ConfigurationSpace()

    d = len(data_points[0])
    gt_cls_num = len(np.unique(labels)) # use ground truth of batch

    if method == "clustream":
        clustream_timewindow = Categorical("time_window",
                                           [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000],
                                           default=batch)
        clustream_mc_r_factor = Float("mc_r_factor", (1.0, 5.0), default=2.0)
        clustream_mmc = Constant("mmc", 100)
        config_space.add([clustream_timewindow, clustream_mc_r_factor, clustream_mmc])
    elif method == "clustream_online":
        clustream_timewindow = Categorical("time_window",
                                           [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000],
                                           default=batch)
        clustream_mc_r_factor = Float("mc_r_factor", (1.0, 5.0), default=2.0)
        clustream_mmc = Integer("mmc", (1, 100), default=gt_cls_num)
        config_space.add([clustream_timewindow, clustream_mc_r_factor, clustream_mmc])
    else:
        raise NotImplementedError(method)
    return config_space