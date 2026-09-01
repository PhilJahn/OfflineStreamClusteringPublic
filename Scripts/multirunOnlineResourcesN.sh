#!/bin/sh
# script thanks to Mamdouh

online_algs='batch_random batch_kcentroid batch_lwc_square batch_protras window_random window_kcentroid window_lwc_square window_protras clustream_river denstream_river streamkmeans_river emcstream_original gbfuzzystream_original birch_sklearn_2 stmstream dstream streamkmplus swclustering'

ns='5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000'

for o in $online_algs; do
	for n in $ns; do
		sbatch --output="./process_log/resource_eval/"$o"_n"$n"_d2_k100_0.txt" ./runResourceEval.sh $o $n 2 100 0
		sbatch --output="./process_log/resource_eval/"$o"_n"$n"_d2_k100_1.txt" ./runResourceEval.sh $o $n 2 100 1
		sbatch --output="./process_log/resource_eval/"$o"_n"$n"_d2_k100_2.txt" ./runResourceEval.sh $o $n 2 100 2
		sbatch --output="./process_log/resource_eval/"$o"_n"$n"_d2_k100_3.txt" ./runResourceEval.sh $o $n 2 100 3
		sbatch --output="./process_log/resource_eval/"$o"_n"$n"_d2_k100_4.txt" ./runResourceEval.sh $o $n 2 100 4
	done
done