#!/bin/bash
#domain="SGD"
#task_set="Banks_1 Buses_1 Buses_2 Calendar_1 Events_1 Events_2 Flights_1 Flights_2 Homes_1 Hotels_1 Hotels_2 Hotels_3 Media_1 Movies_1 Music_1 Music_2 RentalCars_1 RentalCars_2 Restaurants_1 RideSharing_1 RideSharing_2 Services_1 Services_2 Services_3"
domain="MultiWOZ"
task_set="Attraction Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel Hotel+Train Restaurant Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train Taxi Train"
#task_set="Attraction+Hotel Restaurant"
graph_algo="CSILP"

EXP_NUM=100
# Param set to search
#
fgam_set=(0.95)
bgam_set=(0.9)
cneg_set=(4)
pos_set=(30 60 90 120 200 300)
#
dep_set=(6)
leaf_set=(0.01)
###### sequentially run
count=0
parallel_num=40

for task in $task_set; do
  for fgam in ${fgam_set[@]}; do
    for bgam in ${bgam_set[@]}; do
      for cneg in ${cneg_set[@]}; do
        for pos in ${pos_set[@]}; do
          for dep in ${dep_set[@]}; do
            for leaf in ${leaf_set[@]}; do
              echo "count=$count @ Running $task @ $pred_dir"
            
              python run_graph_prediction.py \
              --data_path=../datasets/${domain}/trajectories/${task}_trajectories.json --algo=${graph_algo} \
              --dataset=${domain} --task=${task} --output_dir=../graphs/${domain}/${graph_algo} \
              --forward_discount=${fgam} --backward_discount=${bgam} --negative_horizon=${cneg} --positive_bias=${pos} \
              --max_depth=${dep} --min_leaf_frac=${leaf} &
              count=$(( $count + 1 ))
              if [ $(( count%parallel_num )) -eq 0 ]; then
                  echo "waiting for previous experiments..."
                  wait
              fi
            done
          done
        done
      done
    done
  done
done
wait
echo "finished running $count experiments"