# Repository for Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering

This repository contains the code for the VLDB 2027 submission "Moving Off the Online Phase: Analyzing Summaries for Offline Phase Stream Clustering [Experiment, Analysis & Benchmark]" by Philipp Jahn, Walid Durani, Collin Leiber and Thomas Seidl

## Online Phase

The online phase is controlled through ```online_handler.py```. It defines which online algorithm is used.

The configurations are set in ```online_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_online.py```.


| **Parameter**   | **Function**                                                                                 |
|-----------------|----------------------------------------------------------------------------------------------|
| --ds            | Dataset                                                                                      |
| --online        | Online Method                                                                                |
| --batch         | Evaluation batch size                                                                        |
| --batch_budget  | Duration in seconds per batch (Default: 300_logdim [300 seconds $\cdot log_{10}$ of d])      |
| --smac_seed     | Seed for SMAC                                                                                |
| --summary_limit | Limit on Summary structures (Default: 15000)                                                 |
| --memory_limit  | Memory Limit per configuration in MB (Default: 8000)                                         |
| --trial_limit   | Maximum number of trials (Default: 1000)                                                     |
| --eval_size     | Number of preceding batches to include in evaluation (Default: 1 [2 batches for evaluation]) |

Supported Online Methods:

| **Key(s)**              | **Method**          | 
|-------------------------|---------------------|
| clustream_river         | CluStream           |
| denstream_river         | DenStream           |
| streamkmeans_river      | StreamKMeans        |
| birch_sklearn           | BIRCH               |
| gbfuzzystream_original  | GB-FuzzyStream      |
| emcstream_original      | EmCStream           |
| dstream                 | D-Stream            |
| stmstream               | STM-Stream          |
| swclustering            | SWClustering        |
| streamkmplus            | StreamKM++          |
| batch_random            | Random (Batch)      |
| batch_lwc_square        | LWC (Batch)         |
| batch_kcentroid         | k-Centroid (Batch)  |
| batch_protras           | ProTraS (Batch)     |
| window_random           | Random (Window)     |
| window_lwc_square       | LWC (Window)        |
| window_kcentroid        | k-Centroid (Window) |
| window_protras          | ProTraS (Window)    |
| full_batch              | Full Batch          |

Online replication is performed with ``replicate_online.py`` with analogous settings (automatically detects used final configuration from ``online_param_logs/``)
```--smac_seed``` is replaced by ``--seed`` which specifies which seed to perform the replication with.

## Offline Phase

The offline phase is controlled through ```offline_handler.py```. It defines which online algorithm is used.

The configurations are set in ```offline_config_handler.py```. Each hyperparameter needs to be defined separately.

The optimization for the online phase is done with ```optimize_offline.py```. The code will automatically detect and use the best-scoring online configuration corresponding to the settings from ``online_param_logs/``.

| **Parameter**          | **Function**                                                                                                         |
|------------------------|----------------------------------------------------------------------------------------------------------------------|
| --ds                   | Dataset                                                                                                              |
| --online               | Online Method                                                                                                        |
| --offline              | Offline Method                                                                                                       |
| --batch                | Evaluation batch size                                                                                                |
| --online_batch_budget  | Duration in seconds per batch for (completed) online phase (Default: 300_logdim [300 seconds $\cdot log_{10}$ of d]) |
| --offline_batch_budget | Duration in seconds per batch for offline phase  (Default: 600_logdim [600 seconds $\cdot log_{10}$ of d])           |
| --batch_id             | Batch to optimize                                                                                                    |
| --smac_seed            | Seed for SMAC                                                                                                        |
| --memory_limit         | Memory Limit per configuration in MB (Default: 8000)                                                                 |
| --trial_limit          | Maximum number of trials (Default: 1000)                                                                             |
| --eval_size            | Number of preceding batches to include in evaluation (Default: 1 [2 batches for evaluation])                         |

The pure offline optimization is done with ```optimize_pure_offline```

| **Parameter**    | **Function**                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| --ds             | Dataset                                                                                                              |
| --offline        | Offline Method                                                                                                       |
| --batch          | Evaluation batch size                                                                                                |
| --batch_budget   | Duration in seconds per batch for (completed) online phase (Default: 900_logdim [900 seconds $\cdot log_{10}$ of d]) |
| --batch_id       | Batch to optimize                                                                                                    |
| --smac_seed      | Seed for SMAC                                                                                                        |
| --memory_limit   | Memory Limit per configuration in MB (Default: 8000)                                                                 |
| --trial_limit    | Maximum number of trials (Default: 1000)                                                                             |
| --eval_size      | Number of preceding batches to include in evaluation (Default: 1 [2 batches for evaluation])                         |



Supported Offline Methods:

| **Key(s)** | **Method**                                                                            | 
|------------|---------------------------------------------------------------------------------------|
| adjacency  | Adjacency Processing                                                                  |
| kmeans     | k-Means                                                                               |
| dpc        | Density Peak Clustering                                                               |
| dbscan     | DBSCAN                                                                                |
| fdbscan    | (Float) DBSCAN (weights and min-points multiplied by 100), for D-Stream and DenStream |


## Resource Evaluation

The resource usage of stream clustering algorithms can be examined using ``online_resource_eval.py``

| **Parameter** | **Function**                                       |
|---------------|----------------------------------------------------|
| --online      | Online Method                                      |
| --n           | Number of instances in dataset                     |
| --d           | Dimensionality                                     |
| --k           | Number of blobs/summary structures                 |
| --data_seed   | Seed for Blob dataset                              |
| --method_seed | Seed for Online Method                             |
| --overwrite   | Whether to overwrite old evaluation with same name |

Here, we use ```birch_sklearn_2``` for BIRCH, as the initial ``birch_sklearn`` yielded inconsistent steps sizes for the threshold for different dimensionalities.
The results are provided in ```resource_eval/```.

- ``full_mem`` is the maximum memory usage of the online execution without the first batch
- ``full_summary_mem`` is the maximum memory used by the summary (as this is not captured by the incremental evaluation)
- ``full_min_store_mem`` is the maximum memory used by the summary in the form of mini-summaries (as this is not captured by the incremental evaluation)
- ``full_time`` is the time the online execution needed
- ``full_num`` is the number of summary structures

## Datasets

Supported Datasets (full dataset names are provided in the paper):

| **Key**           | **Dataset** | Source                                                                                                                                                        | Directory                    | Files to add to directory                                              |
|-------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|------------------------------------------------------------------------|
| stream_densired   | SDENS       | included, own                                                                                                                                                 | ``data/``                    | ``stream_densired_data.npy``, ``stream_densired_label.npy`` (included) |
| rbf3              | RBF-3       | [Computational Intelligence Group @ UFSCar's data stream repository](https://github.com/CIG-UFSCar/DS_Datasets/tree/master/Synthetic/Non-Stationary/RBF3_40k) | ``data/``                    | ``RBF3_40000.csv``                                                     |
| har               | HAR         | [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)                                                        | ``data/real_uci/HAR/``       | ``X_test.txt``, ``X_train.txt``, ``y_test.txt``, ``X_train.txt``       |
| electricity       | Elec        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``Electricity.csv``                                                    |
| iabrupt           | InsA        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``INSECTS abrupt_balanced.csv``                                        |
| iincremental      | InsI        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``INSECTS incremental_balanced.csv``                                   |
| gassensor         | GasS        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``GasSensorArray.csv``                                                 |
| pendigits         | PenD        | automatic download, original: [UCI repository](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)                            | ``data/uci_download/``       | automatic                                                              |
| starlightcurves   | SLC         | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``StarLightCurves.csv``                                                |
| adult             | Adult       | [Zenodo](https://zenodo.org/records/12533514)                                                                                                                 | ``data/adult/``              | ``adult_test.csv``, ``adult_train.csv``                                |
| ireocurring       | InsR        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``INSECTS incremental-reoccurring_balanced.csv``                       |
| igradual          | InsG        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``INSECTS gradual_balanced.csv``                                       |
| iabruptreocurring | InsAR       | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``INSECTS incremental-abrupt_balanced.csv``                            |
| posture           | Post        | included, orihinal: [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                        | ``data/usp_download/``       | ``Posture.csv`` (included)                                             |
| smartmeter        | Smart       | included, original: [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                        | ``data/usp_download/``       | ``SmartMeter_LADPU.csv`` (included)                                    |
| uwave             | UWave       | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``UWave.csv``                                                          |
| asfault           | Asflt       | included, original: [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                        | ``data/usp_download/``       | ``Asfault.csv`` (included)                                             |
| powersupply       | Power       | included, original: [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                        | ``data/usp_download/``       | ``Powersupply.csv`` (included)                                         |
| outdoor           | OdO         | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``Outdoor.csv``                                                        |
| htru2             | HT2         | automatic download, original: [UCI repository](https://archive.ics.uci.edu/dataset/372/htru2)                                                                 | ``data/uci_download/``       | automatic                                                              |
| cancer            | Canc        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``Cancer.csv``                                                         |
| sepsissurvival    | SSMCR       | automatic download, original: [UCI repository](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records)                              | ``data/uci_download/``       | automatic                                                              |
| mirs              | MIRS        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``MIRS.csv``                                                           |
| yoga              | Yoga        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``Yoga.csv``                                                           |
| noaa              | NOAA        | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``NOAA.csv``                                                           |
| kddcup99          | KDD99       | automatic download, original: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)                          | ``data/sklearn_download/``   | automatic                                                              |
| covertype         | CovT        | automatic download, original: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html)                           | ``data/sklearn_download/``   | automatic                                                              |
| sensorstream      | Sensor      | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``SensorStream.csv``                                                   |
| pokerhand         | Poker       | automatic download, original: [UCI repository](https://archive.ics.uci.edu/dataset/158/poker+hand)                                                            | ``data/uci_download/``       | automatic                                                              |
| heartbeats        | HeaBea      | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``HeartBeats.csv``                                                     |
| rialto            | RiaBT       | [USP DS repository](https://sites.google.com/view/uspdsrepository)                                                                                            | ``data/usp_download/``       | ``Rialto.csv``                                                         |
| phiusiil          | PPURL       | automatic download, original: [UCI repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)                                         | ``data/uci_download/``       | automatic                                                              |

We modified and reuploaded Post (Posture, Reis et. al., 2016), Asflt (Asfault, Souza, V. M. A., 2018), Power (Powersupply, Zhu, 2010) and Smart (SmartMeter LADPU, Souza et. al., 2021). The edits were restricted to adding missing line breaks. The respective latered lines are denoted next to the loading setting in ``datahandler.py``. They were licensed under the following license:

```LICENCE

This repository is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to:

    Share — copy and redistribute the material in any medium or format;

    Adapt — remix, transform, and build upon the material for any purpose, including research and commercial use.

Under the following terms:

    Attribution — appropriate credit must be given to the repository and original authors, a link to the license must be provided, and any modifications must be indicated.
```


## Results

Evaluation is done using ```result_evaluator.py```.

| **Parameter**          | **Function**                                                                                                         |
|------------------------|----------------------------------------------------------------------------------------------------------------------|
| --ds                   | Dataset                                                                                                              |
| --online               | Online Method                                                                                                        |
| --offline              | Offline Method                                                                                                       |
| --batch                | Evaluation batch size                                                                                                |
| --batch_budget_online  | Duration in seconds per batch for (completed) online phase (Default: 300_logdim [300 seconds $\cdot log_{10}$ of d]) |
| --batch_budget_offline | Duration in seconds per batch for offline phase  (Default: 600_logdim [600 seconds $\cdot log_{10}$ of d])           |
| --eval_size            | Number of preceding batches to include in evaluation (Default: 1 [2 batches for evaluation])                         |
| --overwrite            | Whether to overwrite old evaluation with same name                                                                   |

Alternatively, a batch-based evaluation can be done using ```result_evaluator_partial.py``` which can evaluate batch-wise from ``--start_batch`` (inclusive) to ``--end_batch`` (exclusive). The parts can then be combined using ``result_evaluator_partial2full.py``.

The final results for all of our experiments are provided in ``result_evals/``.

The rankings are then extracted with ```result_extractor.py```. These are provided in ``result_extracts/``, which provides score overviews and rankings for all relevant combinations. These are then utilized by the figure and table extractors.


## Reproducibility

We provide our results in the following Zenodo repositories:

- https://zenodo.org/records/22235275 (DOI:https://doi.org/10.5281/zenodo.22235275)
- https://zenodo.org/records/22238050 (DOI:https://doi.org/10.5281/zenodo.22238050)

The content is stored in .zip files and needs to be extracted into folders of the same name. The offline results had to be split into separate .zip-files. The file names need to be split along the ```-``` into subfolders.

The scripts we used to execute our experiments can be found in ``Scripts/``.

Both worker-b and worker-c share the same hardware (AMD EPYC 9454 48-Core Processor). Original batch-running base script thanks to Mamdouh Aljoud.

## Figures / Tables

- Figure 1: Created with draw.io, content created with ``fig1_content.ipynb`` (see ``figure1_content/``)
- Table 1: Manual
- Figure 2: Created with draw.io, content created with ``sampling_figure.py`` (see ``sampling_figure/``)
- Figure 3: Created with ``sampling_figure.py`` (see ``sampling_figure/``)
- Table 2: Created using the output of ``datahandler.py`` (prints to command line)
- Table 3: Manual
- Table 4: Manual
- Table 5: Created with ``hpo_config_table.py`` (result is ``table_hpo_configs.txt``)
- Performance Boxplots (Figures 5, 6, 7 (left side)): Created with ``rank_change.py`` (``--first/--first_score``: origin of arrow, ``--second/--second_score``: actual boxplot, without arrow (i.e., Figure 5): set both groups to the same) (see ``result_figures/``)
- Critical Difference Diagrams (Figure 7 (right side)): Automatically created with ``result_extractor.py`` (see ``cd_figures/``)
- Figure 8: Created with ``offline_analysis.py`` (see ``result_figures/``)
- Table 6: Created with ``correlation_extractor.py`` (result is ``table_spearman.txt``)
- Figure 9: Created with

## External Code

Code for specific methods were originally from:

CluStream, DenStream, StreamKMeans: [River Stream learning repository](https://github.com/online-ml/river)

EmCStream: [EmCStream repository](https://gitlab.com/alaettinzubaroglu/emcstream)

GB-FuzzyStream: [GBFuzzyStream repository](https://github.com/xjnine/GBFuzzyStream/tree/master)

BIRCH: [scikit-learn](https://github.com/scikit-learn/scikit-learn)

SMAC3: [SMAC3 repository](https://github.com/automl/smac3) (our modifications only restrict the to-file logging of SMAC3 to minimize FileIO during optimization)
