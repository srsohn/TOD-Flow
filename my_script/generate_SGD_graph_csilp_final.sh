#!/bin/bash
domain="SGD"
task_set="Banks_1 Buses_1 Buses_2 Calendar_1 Events_1 Events_2 Flights_1 Flights_2 Homes_1 Hotels_1 Hotels_2 Hotels_3 Media_1 Movies_1 Music_1 Music_2 RentalCars_1 RentalCars_2 Restaurants_1 RideSharing_1 RideSharing_2 Services_1 Services_2 Services_3"
graph_algo="CSILP"
HPARAM="--forward_discount=0.8 --backward_discount=0.95 --positive_bias=3.0 \
        --max_depth=12 --min_leaf_frac=0.0003"

for task in $task_set; do
    echo "Running $task @ $pred_dir"

    python run_graph_prediction.py \
    --data_path=../datasets/${domain}/trajectories/${task}_trajectories.json --algo=${graph_algo} \
    --dataset=${domain} --task=${task} --output_dir=../graphs/${domain}/${graph_algo} $HPARAM
done