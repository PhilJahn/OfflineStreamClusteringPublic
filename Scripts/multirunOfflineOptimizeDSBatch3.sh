#!/bin/sh
# script thanks to Mamdouh

data='kddcup99 covertype sensorstream pokerhand heartbeats rialto phiusiil'
seeds='0 1 2'
online_algs='full_batch batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn stmstream dstream streamkmplus swclustering'
for d in $data; do
	for o1 in $online_algs; do
		for s in $seeds; do
			sbatch --output="./process_log/offline_opt/"$1"_for_"$o1"_on_"$d"_600_logdim_with_1000_"$s".txt" ./runOfflineOptimize600logAll.sh $d $o1 $1 $s
		done
	done
done



