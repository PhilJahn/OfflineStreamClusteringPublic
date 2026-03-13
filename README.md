# Repository for Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering

This repository contains the code for the paper submission "Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering" for ECML PKDD 2026

## Online Phase

The online phase is controlled through ```online_handler.py```. It defines which online algorithm is used.

The configurations are set in ```online_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_online.py```.

## Offline Phase

The offline phase is controlled through ```offline_handler.py```. It defines which online algorithm is used.

The configurations are set in ```offline_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_offline.py```.

## Datasets

The Stream-DENSIRED data is uploaded and was created from data made with [DENSIRED](https://github.com/PhilJahn/DENSIRED)

The Insects Abrupt dataset was taken from the [USP DS repository](https://sites.google.com/view/uspdsrepository)

The RBF3 dataset was taken from the [Computational Intelligence Group @ UFSCar's data stream repository](https://github.com/CIG-UFSCar/DS_Datasets) 

The HAR dataset was taken from the [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The Electricity dataset gets automatically downloaded from [CapyMOA](https://github.com/adaptive-machine-learning/CapyMOA)

## External Code

Code for specific methods were originally from:

CluStream, DenStream, StreamKMeans: [River Stream learning repository](https://github.com/online-ml/river)

EMCStream: [EMCStream repository](https://gitlab.com/alaettinzubaroglu/emcstream)

GB-FuzzyStream: [GBFuzzyStream repository](https://github.com/xjnine/GBFuzzyStream/tree/master)

BIRCH: [scikit-learn](https://github.com/scikit-learn/scikit-learn)

Original CMM Implementation: [Going Offline repository](https://github.com/PhilJahn/SCOPE/)
