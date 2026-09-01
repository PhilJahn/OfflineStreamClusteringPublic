#!/bin/sh
# script thanks to Mamdouh

data='sensorstream'
seeds='0 1 2'
offline_algs='kmeans dbscan dpc'
for d in $data; do
	for o in $offline_algs; do
		for s in $seeds; do
			sbatch --output="./process_log/pure_offline_opt/pure_"$o"_on_"$d"_900_logdim_with_1000_"$s".txt" ./runPureOfflineOptimize900log.sh $d $o $s
		done
	done
done



