#!/bin/sh
: '
TASKS="Banks_1 Buses_1 Buses_2 Calendar_1 Events_1 Events_2 Flights_1 Flights_2 Homes_1 Hotels_1 Hotels_2 Hotels_3 Media_1 Movies_1 Music_1 Music_2 RentalCars_1 RentalCars_2 Restaurants_1 RideSharing_1 RideSharing_2 Services_1 Services_2 Services_3"

for task in $TASKS; do
    echo "task= $task"
    python GPT/demo_trajectory_sampler.py --traj_path="../datasets/SGD/trajectories/${task}_trajectories.json"
done
'

#TASKS="Attraction Hotel Restaurant Taxi Train"
TASKS="Attraction Hotel Restaurant Taxi Train Attraction+Hotel Attraction+Restaurant Attraction+Restaurant+Taxi Attraction+Taxi+Hotel Attraction+Train Hotel+Train Restaurant+Hotel Restaurant+Taxi+Hotel Restaurant+Train"

for task in $TASKS; do
    echo "task= $task"
    python GPT/demo_trajectory_sampler.py --traj_path="../datasets/MultiWOZ_full/trajectories/${task}_trajectories.json"
done
