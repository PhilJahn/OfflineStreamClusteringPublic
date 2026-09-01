#!/bin/sh
# script thanks to Mamdouh

data='starlightcurves adult iincremental ireocurring igradual iabruptreocurring'
online_algs='full_batch batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn stmstream dstream streamkmplus swclustering'
offline_algs='online'
for d in $data; do
	for o1 in $online_algs; do
		for o2 in $offline_algs; do
			sbatch --output="./process_log/eval_online/eval_"$o1"_"$o2"_on_"$d".txt" ./runEvaluate.sh $d $o1 $o2 300_logdim 600_logdim
		done
	done
done



