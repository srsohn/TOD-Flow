#!/bin/bash
domain="MultiWOZ"
#task_set="Attraction Hotel Restaurant Taxi Train"
task_set="Attraction Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel Hotel+Train Restaurant Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train Taxi Train"
graph_algo="SHDILP"
HPARAM=" --min_score=0.95 --complexity_penalty=0.01 --beam_width=4 --beam_depth=4"

for task in $task_set; do
    echo "Running $task @ $pred_dir"

    python run_graph_prediction.py \
    --data_path=../datasets/${domain}/trajectories/${task}_trajectories.json --algo=${graph_algo} \
    --dataset=${domain} --task=${task} --output_dir=../graphs/${domain}/${graph_algo} $HPARAM &
done
wait
echo "experiment done"