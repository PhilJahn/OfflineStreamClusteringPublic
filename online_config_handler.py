import math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition, GreaterThanCondition

from utils import get_data_properties


batch_sampling_methods = ["batch_random", "batch_kcentroid", "batch_lwc_square", "batch_protras"]
window_sampling_methods = ["window_random", "window_kcentroid", "window_lwc_square", "window_protras"]


def get_online_configspace(method, data_points, labels, args):
    config_space = ConfigurationSpace()

    base_1 = Constant("ds", args.ds)
    base_2 = Constant("online", method)
    base_3 = Constant("offline", args.offline)
    base_4 = Constant("batch_size", args.batch)
    base_5 = Constant("eval_size", args.eval_size)
    base_6 = Constant("batch_budget", args.batch_budget)

    config_space.add([base_1, base_2, base_3, base_4, base_5, base_6])

    window_sizes = [100, 200, 250, 500, 1000]

    d, gt_cls_num = get_data_properties(data_points, labels)

    if method == "full_batch":
        return config_space

    if method == "clustream":

        #"max_mc": 100, "init_num": None, "mbf": 2.0, "m": 10, "delta": 512

        clustream_1 = Float("mbf", (1.0, 5.0), default=2.0)
        if args.offline is None:
            clustream_2 = Constant("mmc", 100)
        else:
            clustream_2 = Integer("mmc", (1, 100), default=gt_cls_num)
        clustream_3 = Integer("m", (1, 100), default=10)
        clustream_4 = Integer("delta", (100, 2*args.batch), default=512)
        clustream_5 = Constant("init_num", args.batch)


        config_space.add([clustream_1, clustream_2, clustream_3, clustream_4, clustream_5])
    # elif method == "denstream":
    #     denstream_1 = Constant("init_num", min(args.batch, 1000))
    #     denstream_2 = Integer("mu", (1, 100), default=10)
    #     denstream_3 = Float("beta", (0, 1), default=0.2)
    #     denstream_4 = Float("epsilon", (0, (d**0.5)/2),
    #                         default=0.16)  # assume division by 100 compared to original data (16 -> 0.16)
    #     denstream_5 = Constant("b", 2)  # fixed to 2 since not actually HP in original
    #     denstream_6 = Float("l", (0.01, 1), default=0.25)
    #     denstream_7 = Categorical("stream_speed", window_sizes, default=args.batch)
    #     denstream_8 = Categorical("heuristic", [0, 1], default=1)
    #     denstream_9 = Float("alpha", (0, 1), default=0.5)
    #     config_space.add([denstream_1, denstream_2, denstream_3, denstream_4, denstream_5, denstream_6, denstream_7,
    #                       denstream_8, denstream_9])
    #     denstream_cond_1 = EqualsCondition(config_space['alpha'], config_space['heuristic'], 1)
    #     config_space.add(denstream_cond_1)
    # elif method == "hddstream":
    #     hddstream_1 = Constant("init_num", min(args.batch, 2000))
    #     hddstream_2 = Integer("mu", (1, 100), default=10)
    #     hddstream_3 = Float("beta", (0, 1), default=0.5)
    #     hddstream_4 = Float("epsilon", (0, 1), default=0.2)
    #     hddstream_5 = Constant("b", 2)  # fixed to 2 since not actually HP in original
    #     hddstream_6 = Float("l", (0.01, 1), default=0.5)
    #     hddstream_7 = Categorical("window", window_sizes, default=args.batch)
    #     hddstream_8 = Float("delta", (0, 0.01), default=0.001)
    #     if d - 1 > 1:
    #         hddstream_9 = Integer("pi", (1, d - 1), default=d - 1)
    #     else:
    #         hddstream_9 = Constant("pi", 1)
    #     hddstream_10 = Categorical("kappa", [2, 5, 10, 50, 100, 500, 1000], default=1000)
    #     config_space.add([hddstream_1, hddstream_2, hddstream_3, hddstream_4, hddstream_5, hddstream_6, hddstream_7,
    #                       hddstream_8, hddstream_9, hddstream_10])
    # elif method == "streamkmeans":
    #     streamkmeans_1 = Categorical("k", [10, 25, 50, 100], default=100)
    #     streamkmeans_2 = Categorical("chunk_size",
    #                                  [10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000, 10000],
    #                                  #https://en.wikipedia.org/wiki/Table_of_divisors
    #                                  default=1000)
    #     if args.offline is None:
    #         streamkmeans_3 = Constant("prototype_num", 100)
    #     else:
    #         streamkmeans_3 = Integer("prototype_num", (10, 100), default=max(gt_cls_num,10))
    #         streamkmeans_1 = Categorical("k", [10, 25, 50, 100], default=10)
    #
    #     config_space.add([streamkmeans_1, streamkmeans_2, streamkmeans_3])
    #     streamkmeans_cond_1 = ForbiddenLessThanRelation(config_space['chunk_size'], config_space['k'])
    #     streamkmeans_cond_2 = ForbiddenLessThanRelation(config_space['prototype_num'], config_space['k'])
    #     streamkmeans_cond_3 = ForbiddenGreaterThanRelation(config_space['chunk_size'], config_space['batch_size'])
    #     config_space.add(streamkmeans_cond_1)
    #     config_space.add(streamkmeans_cond_2)
    #     config_space.add(streamkmeans_cond_3)
    # elif method == "streamkmeans+":
    #     streamkmeans_2 = Categorical("chunk_size",
    #                                  [10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000, 10000],
    #                                  #https://en.wikipedia.org/wiki/Table_of_divisors
    #                                  default=1000)
    #     if args.offline is None:
    #         streamkmeans_3 = Constant("prototype_num", 100)
    #         streamkmeans_1 = Integer("k", (2, 100), default=100)
    #     else:
    #         streamkmeans_3 = Integer("prototype_num", (2, 100), default=gt_cls_num)
    #         streamkmeans_1 = Integer("k", (2, 100), default=gt_cls_num)
    #
    #     config_space.add([streamkmeans_1, streamkmeans_2, streamkmeans_3])
    #     streamkmeans_cond_1 = ForbiddenLessThanRelation(config_space['chunk_size'], config_space['k'])
    #     streamkmeans_cond_2 = ForbiddenLessThanRelation(config_space['prototype_num'], config_space['k'])
    #     streamkmeans_cond_3 = ForbiddenGreaterThanRelation(config_space['chunk_size'], config_space['batch_size'])
    #     config_space.add(streamkmeans_cond_1)
    #     config_space.add(streamkmeans_cond_2)
    #     config_space.add(streamkmeans_cond_3)
    elif method == "streamkmplus":
        if args.offline is None:
            streamkmplus_1 = Constant("m", 100)
        else:
            streamkmplus_1 = Integer("m", (1, 100), default=gt_cls_num)
        config_space.add(streamkmplus_1)
    elif method == "dstream":
        #param_dict = {"dim_grid_num": 30, "dim": 2, "value_ranges": 1, "cm": 3.0, "cl": 0.8, "window": 1, "l": 0.998,
        #             "beta": 0.3}
        dstream_1 = Integer("dim_grid_num", (2, 100), default=30)
        dstream_2 = Constant("dim", d)
        dstream_3 = Constant("value_ranges", 1)
        dstream_4 = Float("cm", (1, 10), default=3)
        dstream_5 = Float("cl", (0.01, 1), default=0.8)
        dstream_6 = Constant("window", 1)
        dstream_7 = Float("l", (0.00001,0.5), default=0.002, log=True)
        dstream_8 = Float("beta", (0.001,2), default=0.3)
        config_space.add([dstream_1, dstream_2, dstream_3, dstream_4, dstream_5, dstream_6, dstream_7, dstream_8])
    elif method == "emcstream":
        emcstream_1 = Integer("k", (1, 100), default=gt_cls_num)
        emcstream_2 = Constant("horizon", args.batch)
        emcstream_3 = Integer("init_num_factor", (1,3), default=2)
        emcstream_4 = Integer("drift_num_factor", (1,10), default=5)
        emcstream_5 = Constant("drift_num_factor_change", 1)
        emcstream_6 = Integer("drift_inc_delay", (1,10), default=4)
        emcstream_7 = Constant("consistency_threshold", 1.0)
        emcstream_8 = Float("consistency_drift", (0, 0.1), default=0.001)
        emcstream_9 = Integer("consistency_delay", (1, 10), default=4)
        config_space.add([emcstream_1, emcstream_2, emcstream_3, emcstream_4, emcstream_5, emcstream_6, emcstream_7,
                          emcstream_8, emcstream_9])
        emcstream_cond_1 = ForbiddenLessThanRelation(config_space['drift_num_factor'], config_space['init_num_factor'])
        config_space.add(emcstream_cond_1)
    # elif method == "gbfuzzystream":
    #     gbfuzzystream_1 = Integer("eps", (1, 10), default=10)
    #     gbfuzzystream_2 = Float("m", (1.001, 5), default=2)
    #     gbfuzzystream_3 = Float("l", (0.01, 2), default=1)
    #     gbfuzzystream_4 = Float("threshold", (0.05, 1), default=0.8)
    #     gbfuzzystream_5 = Categorical("window", window_sizes, default=args.batch)
    #     gbfuzzystream_6 = Constant("min_split_size", 8)
    #     config_space.add([gbfuzzystream_1, gbfuzzystream_2, gbfuzzystream_3, gbfuzzystream_4, gbfuzzystream_5, gbfuzzystream_6])
    elif method == "stmstream":
        stmstream_1 = Float("epsilon", (0, 1), default=0.1)
        if args.offline is None:
            stmstream_2 = Constant("k", 100)
        else:
            stmstream_2 = Integer("k", (1, 100), default=gt_cls_num)
        stmstream_3 = Constant("init_num", min(args.batch, 1000))
        stmstream_4 = Categorical("stride", window_sizes, default=args.batch/2)
        stmstream_5 = Categorical("window", window_sizes, default=args.batch)
        stmstream_6 = Float("u_radius", (0.001,0.1), default=0.05)
        config_space.add([stmstream_1, stmstream_2, stmstream_3, stmstream_4, stmstream_5, stmstream_6])
        stmstream_cond_1 = ForbiddenLessThanRelation(config_space['window'], config_space['stride'])
        config_space.add(stmstream_cond_1)
    # elif method == "birch":
    #     if args.offline is None:
    #         birch_1 = Constant("max_leaf_entry_num", 100)
    #     else:
    #         birch_1 = Integer("max_leaf_entry_num", (1, 100), default=gt_cls_num)
    #     birch_2 = Integer("L", (2, 10), default=4) # needs to be 2+ as expansion only compares within leaf -> no additional leafs, no expansion
    #     birch_3 = Integer("B", (1, 4), default=2)
    #     birch_4 = Categorical("closeness", ["D0", "D1", "D2", "D3", "D4"], default="D2")
    #     birch_5 = Categorical("absorption", ["R", "D", "euclidean"], default="D")
    #     birch_6 = Float("outlier_ratio", (0.01, 0.5), default=0.25)
    #     birch_7 = Constant("outlier_batch", 100)
    #     birch_8 = Categorical("mode", ["betula_median", "betula_mean", "birch_greedy", "birch_max"], default="birch_max")
    #
    #     config_space.add([birch_1, birch_2, birch_3, birch_4, birch_5, birch_6, birch_7, birch_8])
    elif method == "swclustering":
        swclustering_1 = Float("epsilon", (0.001, 0.5), default=0.1)
        swclustering_2 = Float("beta", (1.0, 5.0), default=2.0) # same as mbf
        if args.offline is None:
            swclustering_3 = Constant("mmc", 100)
        else:
            swclustering_3 = Integer("mmc", (1, 100), default=gt_cls_num)
        swclustering_4 = Integer("N", (1, 100), default=10)
        config_space.add([swclustering_1, swclustering_2, swclustering_3, swclustering_4])
    elif method in batch_sampling_methods:
        if args.offline is None:
            sampling1 = Constant("sampling_size", 100)
        else:
            sampling1 = Integer("sampling_size", (1, 100), default=gt_cls_num)
        config_space.add([sampling1])
    elif method in window_sampling_methods:
        if args.offline is None:
            sampling1 = Constant("sampling_size", 100)
        else:
            sampling1 = Integer("sampling_size", (1, 100), default=gt_cls_num)

        sampling2 = Float("selection_chance", (0, 1), default=0.5)
        sampling3 = Constant("stride", args.batch) # does not need to be run other than when offline phase is called
        sampling4 = Categorical("window", window_sizes, default=args.batch)
        config_space.add([sampling1, sampling2, sampling3, sampling4])
    # River
    elif method == "clustream_river":
        river_clustream_1 = Float("mbf", (1.0, 5.0), default=2.0)
        if args.offline is None:
            river_clustream_2 = Constant("mmc", 100)
        else:
            river_clustream_2 = Integer("mmc", (1, 100), default=gt_cls_num)
        river_clustream_3 = Integer("window", (100, 2*args.batch), default=512)
        config_space.add([river_clustream_1, river_clustream_2, river_clustream_3])
    elif method == "denstream_river":
        denstream_1 = Constant("init_num", min(args.batch, 1000))
        denstream_2 = Integer("mu", (1, 100), default=10)
        denstream_3 = Float("beta", (0, 1), default=0.2)
        denstream_4 = Float("epsilon", (0, (d**0.5)/2),
                            default=0.16)  # assume division by 100 compared to original data (16 -> 0.16)
        denstream_5 = Float("l", (0.01, 1)
                            , default=0.25)
        denstream_6 = Categorical("stream_speed", window_sizes, default=args.batch)
        config_space.add([denstream_1, denstream_2, denstream_3, denstream_4, denstream_5, denstream_6])
    elif method == "streamkmeans_river":
        streamkmeans_2 = Categorical("chunk_size",
                                     [10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000, 10000],
                                     #https://en.wikipedia.org/wiki/Table_of_divisors
                                     default=1000)
        if args.offline is None:
            streamkmeans_1 = Constant("prototype_num", 100)
        else:
            streamkmeans_1 = Integer("prototype_num", (2, 100), default=gt_cls_num)

        streamkmeans_3 = Float("sigma", (0, 1), default=0.5)
        streamkmeans_4 = Float("mu", (0, 1), default=0.5)

        config_space.add([streamkmeans_1, streamkmeans_2, streamkmeans_3, streamkmeans_4])
        streamkmeans_cond_1 = ForbiddenLessThanRelation(config_space['chunk_size'], config_space['prototype_num'])
        streamkmeans_cond_2 = ForbiddenGreaterThanRelation(config_space['chunk_size'], config_space['batch_size'])
        config_space.add(streamkmeans_cond_1)
        config_space.add(streamkmeans_cond_2)
    elif method == "emcstream_original" or  method == "emcstream_mod":
        emcstream_1 = Integer("k", (1, 100), default=gt_cls_num)
        emcstream_2 = Constant("horizon", args.batch)
        emcstream_3 = Integer("init_num_factor", (1,5), default=2)
        emcstream_4 = Integer("drift_num_factor", (1,10), default=5)
        emcstream_5 = Constant("drift_num_factor_change", 1)
        emcstream_6 = Integer("drift_inc_delay", (1,10), default=4)
        emcstream_7 = Constant("consistency_threshold", 1.0)
        emcstream_8 = Float("consistency_drift", (0, 0.1), default=0.001)
        emcstream_9 = Integer("consistency_delay", (1, 10), default=4)
        config_space.add([emcstream_1, emcstream_2, emcstream_3, emcstream_4, emcstream_5, emcstream_6, emcstream_7,
                          emcstream_8, emcstream_9])
        #emcstream_cond_1 = ForbiddenLessThanRelation(config_space['drift_num_factor'], config_space['init_num_factor'])
        #config_space.add(emcstream_cond_1)
    elif method == "gbfuzzystream_original":
        gbfuzzystream_1 = Integer("eps", (1, 10), default=10)
        gbfuzzystream_2 = Float("m", (1.001, 5), default=2)
        gbfuzzystream_3 = Float("l", (0.01, 2), default=1)
        gbfuzzystream_4 = Float("threshold", (0.05, 1), default=0.8)
        gbfuzzystream_5 = Categorical("window", window_sizes, default=args.batch)
        config_space.add([gbfuzzystream_1, gbfuzzystream_2, gbfuzzystream_3, gbfuzzystream_4, gbfuzzystream_5])
    elif method == "birch_sklearn":
        birch_1 = Float("threshold", (0, (d**0.5)/2), default=0.000000000001)
        birch_2 = Integer("branching_factor", (2, 50), default=50)
        birch_3 = Constant("max_leaf_entry_num", 100)
        config_space.add([birch_1, birch_2, birch_3])
    else:
        raise NotImplementedError(method)
    return config_space
