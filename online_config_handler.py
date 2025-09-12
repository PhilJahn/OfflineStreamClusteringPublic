import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition

from utils import get_data_properties


def get_online_configspace(method, data_points, labels, args):
    config_space = ConfigurationSpace()

    base_1 = Constant("ds", args.ds)
    base_2 = Constant("online", method)
    base_3 = Constant("offline", args.offline)
    base_4 = Constant("batch_size", args.batch)
    base_5 = Constant("batch_max", args.batch_max)
    base_6 = Constant("batch_budget", args.batch_budget)

    config_space.add([base_1, base_2, base_3, base_4, base_5, base_6])

    d, gt_cls_num = get_data_properties(data_points, labels)

    if method == "clustream":
        clustream_1 = Categorical("time_window",
                                           [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000],
                                           default=args.batch)
        clustream_2 = Float("mc_r_factor", (1.0, 5.0), default=2.0)
        if args.offline is None:
            clustream_3 = Constant("mmc", 100)
        else:
            clustream_3 = Integer("mmc", (1, 100), default=gt_cls_num)
        config_space.add([clustream_1, clustream_2, clustream_3])
    else:
        raise NotImplementedError(method)
    return config_space