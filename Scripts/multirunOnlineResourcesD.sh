#!/bin/sh
# script thanks to Mamdouh

online_algs='batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn_2 stmstream dstream streamkmplus swclustering'

ds='5 10 20 50 100 200 500 1000 2000 5000'

for o in $online_algs; do
	for d in $ds; do
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d"$d"_k100_0.txt" ./runResourceEval.sh $o 50000 $d 100 0
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d"$d"_k100_1.txt" ./runResourceEval.sh $o 50000 $d 100 1
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d"$d"_k100_2.txt" ./runResourceEval.sh $o 50000 $d 100 2
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d"$d"_k100_3.txt" ./runResourceEval.sh $o 50000 $d 100 3
		sbatch --output="./process_log/resource_eval/"$o"_n50000_d"$d"_k100_4.txt" ./runResourceEval.sh $o 50000 $d 100 4
	done
done