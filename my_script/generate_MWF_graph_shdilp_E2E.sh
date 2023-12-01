#!/bin/bash
domain="MultiWOZ"
#task_set="Attraction Hotel Restaurant Taxi Train"
task_set="Attraction Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel Hotel+Train Restaurant Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train Taxi Train"
mins_set="0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95"
graph_algo="SHDILP"

HPARAM="--complexity_penalty=0.01 --beam_width=4 --beam_depth=4"

for task in $task_set; do
    for mins in $mins_set; do
        echo "Running $task @ $pred_dir"
    
        python run_graph_prediction.py \
        --data_path=../datasets/${domain}/trajectories/${task}_trajectories.json --algo=${graph_algo} \
        --min_score=${mins} --dataset=${domain} --task=${task} --output_dir=../graphs/${domain}/${graph_algo} $HPARAM
    done
done
wait
echo "experiment done"
