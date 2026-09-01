#!/bin/sh
# script thanks to Mamdouh

online_algs='batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn_2 stmstream dstream streamkmplus swclustering'

ks='2 5 10 20 50 200 500 1000'

for o in $online_algs; do
	for k in $ks; do
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d2_k"$k"_0.txt" ./runResourceEval.sh $o 50000 2 $k 0
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d2_k"$k"_1.txt" ./runResourceEval.sh $o 50000 2 $k 1
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d2_k"$k"_2.txt" ./runResourceEval.sh $o 50000 2 $k 2
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d2_k"$k"_3.txt" ./runResourceEval.sh $o 50000 2 $k 3
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d2_k"$k"_4.txt" ./runResourceEval.sh $o 50000 2 $k 4
	done
done