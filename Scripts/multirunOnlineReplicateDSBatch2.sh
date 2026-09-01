#!/bin/sh
# script thanks to Mamdouh

data='posture smartmeter uwave asfault powersupply outdoor gassensor htru2 cancer sepsissurvival electricity mirs yoga noaa'
seeds='1 2 3 4 5 6 7 8 9'
online_algs='batch_random batch_kcentroid batch_lwc_square window_random window_kcentroid window_lwc_square window_protras streamkmeans_river emcstream_original stmstream streamkmplus'
for d in $data; do
	for o in $online_algs; do
		for s in $seeds; do
			sbatch --output="./process_log/replicate_online_opt/replicate_"$o"_on_"$d"_300_logdim_with_1000_"$s".txt" ./runReplicateOnline.sh $d $o 15000 $s
		done
	done
done



