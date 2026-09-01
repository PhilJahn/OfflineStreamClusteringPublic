#!/bin/sh
# script thanks to Mamdouh

data='posture smartmeter uwave asfault powersupply outdoor gassensor htru2 cancer sepsissurvival electricity mirs yoga noaa'
online_algs='pure_offline'
offline_algs='adjacency kmeans dbscan dpc'
for d in $data; do
	for o1 in $online_algs; do
		for o2 in $offline_algs; do
			sbatch --output="./process_log/eval_pure_"$o2"/eval_"$o1"_"$o2"_on_"$d".txt" ./runEvaluate.sh $d $o1 $o2 900_logdim 600_logdim
		done
	done
done



