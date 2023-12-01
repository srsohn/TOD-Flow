# input: raw data & mapping of (original dataset attributes --> our attributes)
# output: processed dataset (dumped in datasets/{dataset_name}_processed) e.g., MultiWOZ_processed & SGD_processed
# Currently supports 

import json
import os
import importlib
import random
countc = 0
# ------------------Inputs-------------------------------
#sgd_path = '../../dstc8-schema-guided-dialogue/' # the path to the sgd repo folder
#target_json = './hotels1_reservehotel_mapping_v2.json' # the output json path
#task_name = 'Hotels_1_ReserveHotel' # the task name
#schemas = ['Hotels_1'] # schemas to include
#intents = ['ReserveHotel', 'NONE']  # intents to include. Always include 'NONE'
#mappingfile = 'reversible.json' # mapping version in the relabling_mapping subdirectory
#dialog_flow_home_dir = '/home/user/dialogflow' # home directory of dialog_flow 
#-------------------EndOfInput---------------------------
def relabel_dialogs(sgd_path, target_json, task_name, schemas, intents, mappingfile, dialog_flow_home_dir, slotchangedict={}, includeutterandslots=True):
    mapping = json.load(open(dialog_flow_home_dir+'/datasets/SGD/relabling_mapping/'+mappingfile))
    def slotchange(slot):
        if slot in slotchangedict:
            return slotchangedict[slot]
        return slot

    def getslots(frame,utter):
        ret = {}
        for slot in frame['slots']:
            ret[slot['slot']] = utter[slot['start']:slot['exclusive_end']]
        for action in frame['actions']:
            if action['act'] in ['INFORM','CONFIRM','OFFER','SELECT']:
                slot = action['slot']
                if slot not in frame['slots'] and len(action['values']) > 0:
                    ret[slot] = action['values'][0]
        return ret
            


    def parsedialog(dialog):
        turns = dialog['turns']
        turnboxes=[]
        for turn in turns:
            if len(turn['frames'])==1:
                frame = turn['frames'][0]
            else:
                print('ERROR: Frame number unimplemented')
                exit(1)
            actions = frame['actions']
            if 'state' in frame:
                state = frame['state']
            query = None
            if 'service_call' in frame:
                query=(frame['service_call'],frame['service_results'])
            done = []
            options,subtasks = parseutter(turn['speaker'],actions,state,query, done)
            option, subtask = (None, None)
            if query is not None:
                turnboxes.append((turn["utterance"],[parseuttermap(mapping['sys_query'],action='',state=state,query=query,done=done)[0]],'SYSTEM',frame['service_results']))
                call,res=query
                if len(res) == 0:
                    turnboxes.append((turn["utterance"],[parseuttermap(mapping['sys_query_failure'],action='',state=state,query=query, done=done)[1]],'STATUS'))
                    option, subtask=parseuttermap(mapping['sys_query_failure_followup'],action='',state=state,query=query, done=done)
                else:
                    turnboxes.append((turn["utterance"],[parseuttermap(mapping['sys_query_success'],action='',state=state,query=query, done=done)[1]],'STATUS'))
                    option, subtask=parseuttermap(mapping['sys_query_success_followup'],action='',state=state,query=query, done=done)
            if option is not None:
                options.insert(0,option)
            if subtask is not None:
                subtasks.insert(0, subtask)
            turnboxes.append((turn['utterance'],options,turn['speaker'],getslots(frame,turn['utterance'])))
            turnboxes.append((turn['utterance'],subtasks,'STATUS'))
        return turnboxes

    def processmap(mapp, state, query,slot,value,speaker):
        if mapp is None:
            return None
        m = mapp.replace('<SPEAKER>', speaker)
        m = m.replace('<SLOT>',slotchange(str(slot)))
        m = m.replace('<VALUE>', str(value))
        m = m.replace('<CURRENTINTENT>', str(state['active_intent']))
        return m


    def parseuttermap(mapped, action, state, query, slot = '', value = None, speaker = 'USER', done=[]):
        onlyonce = False
        subtask = None
        if 'generalstatus' in mapped: 
            subtask = processmap(mapped['generalstatus'], state=state, query=query, slot=slot,value=value, speaker=speaker)
        else:
            if 'systemstatus' in mapped and speaker == 'SYSTEM':
                subtask = processmap(mapped['systemstatus'], state=state, query=query, slot=slot, value=value, speaker=speaker)
            elif 'userstatus' in mapped and speaker == 'USER':
                subtask = processmap(mapped['userstatus'], state=state, query=query, slot=slot, value=value, speaker=speaker)
        if 'onlyonce' in mapped:
            onlyonce = mapped['onlyonce']
        if onlyonce: 
            if action not in done:
                done.append(action)
            else:
                return None, None
        if 'intentcondition' in mapped:
            if state['active_intent'] != mapped['intentcondition']:
                return None, None
        if 'generalmap' in mapped:
            return processmap(mapped['generalmap'], state=state, query=query, slot=slot,value=value, speaker=speaker),subtask
        else:
            if speaker == 'SYSTEM' and 'systemmap' in mapped:
                return processmap(mapped['systemmap'], state=state, query=query, slot=slot,value=value, speaker=speaker),subtask
            elif 'usermap' in mapped:
                return processmap(mapped['usermap'], state=state, query=query, slot=slot,value=value, speaker=speaker),subtask
            else:
                return None, subtask




    def parseutter(speaker,actions,state,query=None, done = []):
        boxes1=[]
        boxes2=[]
        for action in actions:
            c1,c2 = parseuttermap(mapping[action['act']],action['act'],state,query,action['slot'],action['values'],speaker,done)
            if c1 is not None:
                boxes1.append(c1)
            if c2 is not None:
                boxes2.append(c2)
        return boxes1, boxes2



    def getdialogs(schemas, intents):
        dialogs = []
        for name in os.listdir(sgd_path+'/'+split):
            if name[0:4]=='dial':
                d = json.load(open(sgd_path+'/'+split+'/'+name))
                for dialog in d:
                    if len(dialog['services']) > 1 or dialog['services'][0] not in schemas:
                        continue
                    turns = dialog['turns']
                    good=True
                    for turn in turns:
                        frames = turn['frames']
                        for frame in frames:
                            if 'state' in frame:
                                if frame['state']['active_intent'] not in intents:
                                    #print(frame['state']['active_intent'])
                                    good = False
                    if good:
                        dialogs.append(dialog)
        return dialogs
                                

    dialogs = getdialogs(schemas, intents)
    print(len(dialogs))        
    boxeses = []
    for dialog in dialogs:
        boxes = parsedialog(dialog)
        boxeses.append(boxes)
    allboxes = {}
    for boxes in boxeses:
        for boxs in boxes:
            for box in boxs[1]:
                #print(box)
                allboxes[box] = subtaskoption(box)
    sample = {'option_labels':[],'subtask_labels':[]}
    for box in allboxes:
        sample[allboxes[box]].append(box)
    sample['num_subtask']= len(sample['subtask_labels'])
    sample['num_option']= len(sample['option_labels'])
    trajectories = []
    for i,dialog in enumerate(dialogs):
        traj = {}
        traj['name']=dialog['dialogue_id']
        boxes = boxeses[i]
        soindices = []
        so = []
        for j,utterboxs in enumerate(boxes):
            if utterboxs[2] != 'STATUS':
                if includeutterandslots:
                    if len(utterboxs) > 3:
                        so.append(['option', utterboxs[1],utterboxs[0],utterboxs[3]])
                    else:
                        so.append(['option', utterboxs[1],utterboxs[0]])
                else:
                    so.append(['option',utterboxs[1]])
                soindices.append(['option', [sample['option_labels'].index(x) for x in utterboxs[1]]])
            else:
                if includeutterandslots:
                    so.append(['subtask', utterboxs[1],utterboxs[0]])
                else:
                    so.append(['subtask',utterboxs[1]])
                soindices.append(['subtask', [sample['subtask_labels'].index(x) for x in utterboxs[1]]])
        traj['subtask_and_option_indices'] = soindices
        traj['subtasks_and_options'] = so
        trajectories.append(traj)

    # add train/val splits
    #"""
    valnum = len(trajectories) // 10
    vals = random.sample(trajectories,valnum)
    for traj in trajectories:
        if traj in vals:
            traj['split'] = 'val'
        else:
            traj['split'] = 'train'
    #"""
    sample["trajectories"]=trajectories
    if target_json is not None and len(trajectories) > 0:
        f=open(target_json,'w+')
        json.dump({task_name: sample},f,indent=2)
    else:
        return {task_name: sample}



split = 'train' 
def subtaskoption(box):
    if box[0:6] == 'STATUS':
        return 'subtask_labels'
    else:
        return 'option_labels'

if __name__== '__main__':
    #"""
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Banks_1_trajectories.json', # the output json path
        task_name = 'Banks_1_CheckBalance_TransferMoney', # the task name
        schemas = ['Banks_1'], # schemas to include
        intents = ['CheckBalance', 'TransferMoney', 'NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Flights_1_trajectories.json', # the output json path
        task_name = 'Flights_1_Search_Reserve', # the task name
        schemas = ['Flights_1'], # schemas to include
        intents = ['SearchOnewayFlight','SearchRoundtripFlights', 'ReserveOnewayFlight', 'ReserveRoundtripFlights','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Restaurants_1_trajectories.json', # the output json path
        task_name = 'Restaurants_1_Search_Reserve', # the task name
        schemas = ['Restaurants_1'], # schemas to include
        intents = ['FindRestaurants','ReserveRestaurant','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/RideSharing_1_trajectories.json', # the output json path
        task_name = 'RideSharing_1_GetRide', # the task name
        schemas = ['RideSharing_1'], # schemas to include
        intents = ['GetRide','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Services_1_trajectories.json', # the output json path
        task_name = 'Services_1_Search_Reserve', # the task name
        schemas = ['Services_1'], # schemas to include
        intents = ['BookAppointment','FindProvider','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Music_1_trajectories.json', # the output json path
        task_name = 'Music_1_Find_Play', # the task name
        schemas = ['Music_1'], # schemas to include
        intents = ['LookupSong','PlaySong','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/RentalCars_1_trajectories.json', # the output json path
        task_name = 'RentalCars_1_Search_Reserve', # the task name
        schemas = ['RentalCars_1'], # schemas to include
        intents = ['GetCarsAvailable','ReserveCar','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Buses_1_trajectories.json', # the output json path
        task_name = 'Buses_1_Search_BuyTicket', # the task name
        schemas = ['Buses_1'], # schemas to include
        intents = ['FindBus','BuyBusTicket','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Calendar_1_trajectories.json', # the output json path
        task_name = 'Calendar_1_Get_Add', # the task name
        schemas = ['Calendar_1'], # schemas to include
        intents = ['GetAvailableTime','AddEvent','GetEvents','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Hotels_1_trajectories.json', # the output json path
        task_name = 'Hotels_1_Reserve_Search', # the task name
        schemas = ['Hotels_1'], # schemas to include
        intents = ['ReserveHotel','SearchHotel','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Buses_2_trajectories.json', # the output json path
        task_name = 'Buses_2_Search_BuyTicket', # the task name
        schemas = ['Buses_2'], # schemas to include
        intents = ['FindBus','BuyBusTicket','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Events_1_trajectories.json', # the output json path
        task_name = 'Events_1_Search_BuyTicket', # the task name
        schemas = ['Events_1'], # schemas to include
        intents = ['FindEvents','BuyEventTickets','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Events_2_trajectories.json', # the output json path
        task_name = 'Events_2_Search_BuyTicket', # the task name
        schemas = ['Events_2'], # schemas to include
        intents = ['FindEvents','BuyEventTickets','GetEventDates','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Flights_2_trajectories.json', # the output json path
        task_name = 'Flights_2_Search_BuyTicket', # the task name
        schemas = ['Flights_2'], # schemas to include
        intents = ['SearchOnewayFlight','SearchRoundtripFlights','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Homes_1_trajectories.json', # the output json path
        task_name = 'Homes_1_Search_BuyTicket', # the task name
        schemas = ['Homes_1'], # schemas to include
        intents = ['FindApartment','ScheduleVisit','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Hotels_2_trajectories.json', # the output json path
        task_name = 'Hotels_2_Reserve_Search', # the task name
        schemas = ['Hotels_2'], # schemas to include
        intents = ['BookHouse','SearchHouse','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Hotels_3_trajectories.json', # the output json path
        task_name = 'Hotels_3_Reserve_Search', # the task name
        schemas = ['Hotels_3'], # schemas to include
        intents = ['ReserveHotel','SearchHotel','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Media_1_trajectories.json', # the output json path
        task_name = 'Media_1_Reserve_Search', # the task name
        schemas = ['Media_1'], # schemas to include
        intents = ['FindMovies','PlayMovie','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Movies_1_trajectories.json', # the output json path
        task_name = 'Movies_1_Reserve_Search', # the task name
        schemas = ['Movies_1'], # schemas to include
        intents = ['FindMovies','GetTimesForMovie','BuyMovieTickets','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Music_2_trajectories.json', # the output json path
        task_name = 'Music_2_Reserve_Search', # the task name
        schemas = ['Music_2'], # schemas to include
        intents = ['LookupMusic','PlayMedia','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )

    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/RentalCars_2_trajectories.json', # the output json path
        task_name = 'RentalCars_2_Search_Reserve', # the task name
        schemas = ['RentalCars_2'], # schemas to include
        intents = ['GetCarsAvailable','ReserveCar','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/RideSharing_2_trajectories.json', # the output json path
        task_name = 'RideSharing_2_GetRide', # the task name
        schemas = ['RideSharing_2'], # schemas to include
        intents = ['GetRide','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Services_2_trajectories.json', # the output json path
        task_name = 'Services_2_Search_Reserve', # the task name
        schemas = ['Services_2'], # schemas to include
        intents = ['BookAppointment','FindProvider','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Services_3_trajectories.json', # the output json path
        task_name = 'Services_3_Search_Reserve', # the task name
        schemas = ['Services_3'], # schemas to include
        intents = ['BookAppointment','FindProvider','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Travel_1_trajectories.json', # the output json path
        task_name = 'Travel_1_Search_Reserve', # the task name
        schemas = ['Travel_1'], # schemas to include
        intents = ['FindAttractions','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )
    relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = '../datasets/SGD/trajectories/Weather_1_trajectories.json', # the output json path
        task_name = 'Weather_1_Search_Reserve', # the task name
        schemas = ['Weather_1'], # schemas to include
        intents = ['GetWeather','NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'':'all'}
    )

    """
    h1 = relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = None, # the output json path
        task_name = 'Hotels_1_ReserveHotel', # the task name
        schemas = ['Hotels_1'], # schemas to include
        intents = ['ReserveHotel', 'NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..'
    )
    h2 = relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = None, # the output json path
        task_name = 'Hotels_2_BookHouse', # the task name
        schemas = ['Hotels_2'], # schemas to include
        intents = ['BookHouse', 'NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..'
    )
    h3 = relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = None, # the output json path
        task_name = 'Hotels_3_ReserveHotel', # the task name
        schemas = ['Hotels_3'], # schemas to include
        intents = ['ReserveHotel', 'NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..'
    )
    h4 = relabel_dialogs(
        sgd_path = '../../dstc8-schema-guided-dialogue/', # the path to the sgd repo folder
        target_json = None, # the output json path
        task_name = 'Hotels_13_ReserveHotel_combined', # the task name
        schemas = ['Hotels_3','Hotels_1'], # schemas to include
        intents = ['ReserveHotel', 'NONE'],  # intents to include. Always include 'NONE'
        mappingfile = 'reversible.json', # mapping version in the relabling_mapping subdirectory
        dialog_flow_home_dir = '..',
        slotchangedict = {'destination': 'location', 'price_per_night': 'price'}
    )


    f=open('../outputs/sample_data_reversible.json','w+')
    h1.update(h2)
    h1.update(h3)
    h1.update(h4)
    json.dump(h1,f,indent=2)
    """
