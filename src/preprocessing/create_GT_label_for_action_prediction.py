import json

# create ground truth label json files from trajectories json files
def standardize_trajectory(trajpath,task,outputpath,testonly=True, dataset='SGD'):
    """
    trajpath: path to the trajectories json
    task: name of the task. Can just put None if there is only one task in the trajectories file
    outputpath: path to the output json
    testonly: whether to create labels for the test set or the train+val set.
    dataset: the name of the dataset we are dealing with
    """
    bigjson = json.load(open(trajpath))
    if task is None:
        for i in bigjson.keys():
            task = i
            break
    alltrajs = bigjson[task]['trajectories']
    ret = []
    for traj in alltrajs:
        if traj['split'] != 'test':
            if testonly:
                continue
        else:
            if not testonly:
                continue
        retd = {"dialog_id": traj['name'],"turns":[]}
        turnid = 0
        turn = initturn(turnid)
        speaker = ''
        for a,so in enumerate(traj['subtasks_and_options']):
            s = so[0]
            if s == 'option':
                if dataset == 'SGD':
                    speaker = getspeaker(so[1])
                else:
                    speaker = 'USER'
                    if a % 4 == 2:
                        speaker = 'SYSTEM'
                
                turn[speaker]['action'] = so[1]
            else:
                turn[speaker]['status'] = so[1]
                if speaker.lower() == 'system':
                    retd['turns'].append(turn)
                    turnid += 1
                    turn = initturn(turnid)
        ret.append(retd)
    print(outputpath)
    json.dump(ret,open(outputpath,'w+'),indent=2)

# get the speaker from an action in SGD
def getspeaker(li):
    if li[0][0:4].lower() in ['user','none']:
        return 'USER'
    return 'SYSTEM'

# create an initial turn dictionary from the turn id
def initturn(tid):
    return {'turn_id': tid, "USER":{},"SYSTEM":{}}

import os, sys

# create all SGD gt labels
if sys.argv[1] == 'SGD':
    standardize_trajectory('../datasets/SGD/trajectories/Banks_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Banks_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Buses_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Buses_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Buses_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Buses_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Calendar_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Calendar_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Flights_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Flights_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Flights_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Flights_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Hotels_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Hotels_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_3_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Hotels_3_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Music_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Music_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Music_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Music_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/RentalCars_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/RentalCars_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/RentalCars_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/RentalCars_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Restaurants_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Restaurants_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/RideSharing_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/RideSharing_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/RideSharing_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/RideSharing_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Services_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Services_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Services_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Services_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Services_3_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Services_3_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Homes_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Homes_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Media_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Media_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Movies_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Movies_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Events_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Events_1_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Events_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_train+val/Events_2_labels.json',testonly=False)
    standardize_trajectory('../datasets/SGD/trajectories/Banks_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Banks_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Buses_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Buses_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Buses_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Buses_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Calendar_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Calendar_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Flights_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Flights_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Flights_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Flights_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Hotels_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Hotels_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Hotels_3_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Hotels_3_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Music_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Music_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Music_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Music_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/RentalCars_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/RentalCars_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/RentalCars_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/RentalCars_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Restaurants_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Restaurants_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/RideSharing_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/RideSharing_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/RideSharing_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/RideSharing_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Services_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Services_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Services_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Services_2_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Services_3_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Services_3_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Homes_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Homes_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Media_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Media_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Movies_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Movies_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Events_1_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Events_1_labels.json')
    standardize_trajectory('../datasets/SGD/trajectories/Events_2_trajectories.json',None,'../datasets/SGD/action_prediction_gt_labels_test_only/Events_2_labels.json')


# The following lines are for creating both test-only and train_val labels for MultiWoz
elif sys.argv[1] == 'MultiWOZ':
    for name in os.listdir('../datasets/MultiWOZ/trajectories'):
        if len(name) < 5:
            continue
        standardize_trajectory('../datasets/MultiWOZ/trajectories/'+name,None,'../datasets/MultiWOZ/action_prediction_gt_labels_test_only/'+name.replace('trajectories','labels'),testonly=True, dataset = 'multiwoz')
        standardize_trajectory('../datasets/MultiWOZ/trajectories/'+name,None,'../datasets/MultiWOZ/action_prediction_gt_labels_train+val/'+name.replace('trajectories','labels'),testonly=False, dataset = 'multiwoz')
