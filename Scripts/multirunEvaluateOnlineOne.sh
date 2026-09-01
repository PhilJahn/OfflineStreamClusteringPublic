#!/bin/sh
# script thanks to Mamdouh

data='stream_densired rbf3 har pendigits iabrupt starlightcurves adult iincremental ireocurring igradual iabruptreocurring posture smartmeter uwave asfault powersupply outdoor gassensor htru2 cancer sepsissurvival electricity mirs yoga noaa kddcup99 covertype sensorstream heartbeats rialto phiusiil pokerhand'
online_algs='one'
offline_algs='online'
for d in $data; do
	for o1 in $online_algs; do
		for o2 in $offline_algs; do
			sbatch --output="./process_log/eval_online_seed/eval_"$o1"_"$o2"_on_"$d".txt" ./runEvaluate.sh $d $o1 $o2 300_logdim 600_logdim
		done
	done
done



