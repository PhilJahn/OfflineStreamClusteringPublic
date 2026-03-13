# Repository for Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering

This repository contains the code for the paper submission "Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering" for ECML PKDD 2026

## Online Phase

The online phase is controlled through ```online_handler.py```. It defines which online algorithm is used.

The configurations are set in ```online_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_online.py```.


| **Parameter**  | **Default Value**      | **Function**                  |
|----------------|------------------------|-------------------------------|
| --ds           | stream_densired        | Dataset                       |
| --online       | gbfuzzystream_original | Online Method                 |
| --batch        | 1000                   | Evaluation batch size         |
| --batch_budget | 10                     | Duration in seconds per batch |
| --smac_seed    | 42                     | Seed of SMAC seed             |

Supported Online Methods:

| **Key(s)**                   | **Method**       | 
|------------------------------|------------------|
| clustream_river              | CluStream        |
| denstream_river              | DenStream        |
| streamkmeans_river           | StreamKMeans     |
| birch_sklearn                | BIRCH            |
| gbfuzzystream_original       | GB-FuzzyStream   |
| emcstream_original           | EMCStream        |
| dstream, dstream+, dstream++ | D-Stream         |
| stmstream, stmstream+        | STM-Stream       |
| swclustering, swclustering+  | SWClustering     |
| streamkmplus, streamkmplus+  | StreamKM++       |
| random                       | (Uniform) Random |
| lwc_square                   | LWC              |
| kcentroid                    | k-Centroid       |
| protras                      | ProTraS          |
| full_batch                   | Full Batch       |

## Offline Phase

The offline phase is controlled through ```offline_handler.py```. It defines which online algorithm is used.

The configurations are set in ```offline_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_offline.py```.

| **Parameter**  | **Default Value**      | **Function**                                                                                  |
|----------------|------------------------|-----------------------------------------------------------------------------------------------|
| --ds           | stream_densired        | Dataset                                                                                       |
| --online       | gbfuzzystream_original | Online Method                                                                                 |
| --offline      | dbscan                 | Offline Method                                                                                |
| --batch        | 1000                   | Evaluation batch size                                                                         |
| --batch_budget | 1800                   | Duration in seconds per batch                                                                 |
| --batch_id     | -1                     | Batch to optimize (if set to value, optimizes over chunk of 10 starting with specified batch) |
| --mult         | 0                      | Online summaries optimized with alternate penalty (unused)                                    |
| --smac_seed    | 42                     | Seed of SMAC seed                                                                             |

The pure offline optimization is done with ```optimize_pure_offline```

| **Parameter**  | **Default Value**      | **Function**                                                                                  |
|----------------|------------------------|-----------------------------------------------------------------------------------------------|
| --ds           | stream_densired        | Dataset                                                                                       |
| --offline      | dbscan                 | Offline Method                                                                                |
| --batch        | 1000                   | Evaluation batch size                                                                         |
| --batch_budget | 10                     | Duration in seconds per batch                                                                 |
| --smac_seed    | 42                     | Seed of SMAC seed                                                                             |

Supported Online Methods:

| **Key(s)** | **Method**                                                                            | 
|------------|---------------------------------------------------------------------------------------|
| kmeans     | k-Means                                                                               |
| dpc        | Density Peak Clustering                                                               |
| dbscan     | DBSCAN                                                                                |
| fdbscan    | (Float) DBSCAN (weights and min-points multiplied by 100), for D-Stream and DenStream |

## Datasets

The Stream-DENSIRED data is uploaded and was created from data made with [DENSIRED](https://github.com/PhilJahn/DENSIRED)

The Insects Abrupt dataset was taken from the [USP DS repository](https://sites.google.com/view/uspdsrepository)

The RBF-3 40000 dataset was taken from the [Computational Intelligence Group @ UFSCar's data stream repository](https://github.com/CIG-UFSCar/DS_Datasets) 

The HAR dataset was taken from the [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The Electricity dataset gets automatically downloaded from [CapyMOA](https://github.com/adaptive-machine-learning/CapyMOA)

Supported Datasets Methods:

| **Key(s)**     | **Method**      | 
|----------------|-----------------|
| streamdensired | Stream-DENSIRED |
| insects_abrupt | Insects Abrupt  | 
| rbf3           | RBF-3 40000     | 
| har            | HAR             | 
| electricity    | Electricity     |

## Results

As the results are generally too large to properly store in this repository, we only report the following:

The optimization results are in ```\online_param_logs``` for online, ```\offline_param_logs``` for offline and ```\pure_offline_param_logs``` for pure offline. The folders are split by datasets. In the case of the offline methods they are also split by the online and then offline methods. 
The first line is the default value, the second line the one optained by the optimization. The optimization score (lower is better) is listed next to the dictionary containing the configuration. Next to it are the number of configuration candidates examined and the filepath to the labels (which includes the information which candidate index was selected).

The final results in ```\result_evals```, which contains the individual scores extracted from the experiment results as dictionaries stored in as npy-files.

## External Code

Code for specific methods were originally from:

CluStream, DenStream, StreamKMeans: [River Stream learning repository](https://github.com/online-ml/river)

EMCStream: [EMCStream repository](https://gitlab.com/alaettinzubaroglu/emcstream)

GB-FuzzyStream: [GBFuzzyStream repository](https://github.com/xjnine/GBFuzzyStream/tree/master)

BIRCH: [scikit-learn](https://github.com/scikit-learn/scikit-learn)

Original CMM Implementation: [Going Offline repository](https://github.com/PhilJahn/SCOPE/)
