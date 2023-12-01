#!/bin/bash
domain="MultiWOZ"
#task_set="Attraction Hotel Restaurant Taxi Train"
task_set="Attraction Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel Hotel+Train Restaurant Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train Taxi Train"
graph_algo="CSILP"
HPARAM="--forward_discount=0.2 --backward_discount=0.2 --positive_bias=10 \
        --max_depth=8 --min_leaf_frac=0.015"

for task in $task_set; do
    echo "count=$count @ Running $task @ $pred_dir"

    python run_graph_prediction.py \
    --data_path=../datasets/${domain}/trajectories/${task}_trajectories.json --algo=${graph_algo} \
    --dataset=${domain} --task=${task} --output_dir=../graphs/${domain}/${graph_algo} $HPARAM
done