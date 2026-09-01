#!/bin/sh
# script thanks to Mamdouh

data='stream_densired rbf3 starlightcurves har pendigits adult iabrupt iincremental ireocurring igradual iabruptreocurring'
seeds='0 1 2'
online_algs='full_batch batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn stmstream dstream streamkmplus swclustering'
for d in $data; do
	for o in $online_algs; do
		for s in $seeds; do
			sbatch --output="./process_log/online_opt/"$o"_on_"$d"_300_logdim_with_1000_"$s".txt" ./runOnlineOptimize300log.sh $d $o $s 15000
		done
	done
done



