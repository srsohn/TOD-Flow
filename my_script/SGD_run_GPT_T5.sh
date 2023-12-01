#!/bin/bash
domain="SGD"
task_set="Banks_1 Buses_1 Buses_2 Calendar_1 Events_1 Events_2 Flights_1 Flights_2 Homes_1 Hotels_1 Hotels_2 Hotels_3 Media_1 Movies_1 Music_1 Music_2 RentalCars_1 RentalCars_2 Restaurants_1 RideSharing_1 RideSharing_2 Services_1 Services_2 Services_3"
seed='1636423'
model=$1
temp_set="0 1"

######
for temp in $temp_set; do
    for task in $task_set; do
        pred_dir="../outputs/$domain/${task}_${model}_T${temp}"
        #
        python run_LLM_DM.py \
        --dataset=${domain} --model=${model} --temperature=${temp} \
        --seed=${seed} \
        --traj_path=../datasets/${domain}/trajectories/${task}_trajectories.json \
        --output_dir=${pred_dir}
        #
    done
done