#!/bin/bash
domain="MultiWOZ"
#task_set="Attraction Hotel Restaurant Taxi Train"
task_set="Attraction Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel Hotel+Train Restaurant Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train Taxi Train"
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