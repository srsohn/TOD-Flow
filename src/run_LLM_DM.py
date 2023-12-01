from tqdm import tqdm
import copy
import json
import random
import os
import time
import numpy as np
from absl import flags,app
from util.data_utils import DataLoader
from util.prompt_utils import subtask_options_to_prompt, prompt_to_subtask_options, get_natural_task_name
from util.logic_utils import to_multi_hot
from util.graph_utils import load_gt_graph_sop
from util.demo_utils import fetch_embed, find_closest_in_train, processdemo
from GPT.GPTutils import num_tokens_from_prompt

TASK_SET_BY_DATASET = dict(
    sgd=["Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2", "Flights_1", "Flights_2", "Homes_1", "Hotels_1", "Hotels_2", "Hotels_3", "Media_1", "Movies_1", "Music_1", "Music_2", "RentalCars_1", "RentalCars_2", "Restaurants_1", "RideSharing_1", "RideSharing_2", "Services_1", "Services_2", "Services_3"],
    multiwoz=["Attraction", "Hotel", "Restaurant", "Taxi", "Train", "Attraction+Hotel", "Attraction+Restaurant", "Attraction+Restaurant+Taxi", "Attraction+Taxi+Hotel", "Attraction+Train", "Hotel+Train", "Restaurant+Hotel", "Restaurant+Taxi+Hotel", "Restaurant+Train"]
)

def get_demo_prompt(demo_trajectories, precond, use_mask_prompt, subtask_labels, option_labels, prompt_style=None):
    num_subtask = len(subtask_labels)
    demo_prompt = ""
    if 'entire' in prompt_style:
        for demo_index, demo_trajectory in enumerate(demo_trajectories):
            # Put entire trajectory
            demo_prompt += f"Demonstration:\n"
            demo_prompt += subtask_options_to_prompt(demo_trajectory['entire_subtasks_and_options'], prompt_style)
    elif 'qa' in prompt_style:
        for demo_index, demo_trajectory in enumerate(demo_trajectories):
            history_prompt, system_act_prompt = trajectory_to_prompt(demo_trajectory, prompt_style)
            #
            demo_prompt += f"Demonstration:\n"
            demo_prompt += "Input:\n"
            demo_prompt += history_prompt
            if use_mask_prompt:
                completion = trajectory_to_completion(demo_trajectory, num_subtask)
                action_mask_prompt = get_action_mask_prompt(completion, precond, option_labels, prompt_style)
                demo_prompt += f"Available {action_mask_prompt}"
                if 'symbolic' in prompt_style:
                    demo_prompt += '\n'
            demo_prompt += "Output:\n"
            demo_prompt += system_act_prompt + "\n"
    else:
        raise NotImplementedError
    return demo_prompt

def get_task_prompt(trajectory, turn_index, precond, use_mask_prompt, subtask_labels, option_labels, prompt_style=None):
    num_subtask = len(subtask_labels)
    history_prompt, system_act_prompt = trajectory_to_prompt(trajectory, prompt_style=prompt_style, end=turn_index)
    
    if 'entire' in prompt_style:
        task_prompt = "Based on the demonstrations above, predict the next line of SYSTEM Actions below:\n"
    elif 'qa' in prompt_style:
        task_prompt = f"Task:\n"
        task_prompt += "Input:\n"
    task_prompt += history_prompt
    if use_mask_prompt:
        completion = trajectory_to_completion(trajectory, num_subtask, end=turn_index)
        action_mask_prompt = get_action_mask_prompt(completion, precond, option_labels, prompt_style)
        task_prompt += f"Available {action_mask_prompt}"
        if 'symbolic' in prompt_style:
            task_prompt += '\n'
    if 'entire' in prompt_style:
        task_prompt += system_act_prompt
    elif 'qa' in prompt_style:
        task_prompt += "Output:\n"
        task_prompt += system_act_prompt
    return task_prompt

def trajectory_to_prompt(trajectory, prompt_style, end=None):
    # prompt about history
    history_prompt = subtask_options_to_prompt(trajectory['subtasks_and_options'], prompt_style, end)
    
    # system option prompt
    if 'gt_system_option' in trajectory: # when input 'trajectory' is a demonstration
        system_act_prompt = subtask_options_to_prompt([trajectory['gt_system_option']], prompt_style, end)
    else: # for current task
        if 'concise' in prompt_style:
            system_act_prompt =  'SYSTEM act: '
        elif 'symbolic' in prompt_style:
            system_act_prompt =  'Actions: SYSTEM '
    
    return history_prompt, system_act_prompt

def trajectory_to_completion(trajectory, num_subtask, end=None):
    if end is None:
        subtask_and_option_indices = trajectory['subtask_and_option_indices']
    else:
        subtask_and_option_indices = trajectory['subtask_and_option_indices'][:end]
        
    prev_comp = np.zeros((num_subtask), dtype=bool)
    for turn_index, subtask_and_option_index in enumerate(subtask_and_option_indices):
        turn_type, indices = subtask_and_option_index # 'option', [0, 1, 2]
        if turn_type == 'subtask': # completion
            new_comp = to_multi_hot(np.array(indices), num_subtask)
            curr_comp = prev_comp | new_comp
            prev_comp = curr_comp
        elif turn_type == 'option': # eligibility
            pass
        else:
            assert False, "error in parsing data"
    completion = np.expand_dims(curr_comp, axis=0) # [d] -> [1, d]
    return completion

def get_action_mask_prompt(completion, precond, option_labels, prompt_style):
    from util.logic_utils import predict_elig
    gt_sop, neg_precond = precond
    elig_mask = np.stack([predict_elig(op_sop, completion) for op_sop in gt_sop], axis=1)[0]
    neg_precond_mat = np.array(neg_precond['precondition_vectors'])
    nonelig_mask = neg_precond_mat @ completion.T # (#option, 1)
    label_list = []
    for option_index, elig in enumerate(elig_mask):
        if elig and 'SYSTEM' in option_labels[option_index] and not nonelig_mask[option_index]:
            label_list.append(option_labels[option_index])
    subtasks_and_options = [["option", label_list]]
    action_mask_prompt = subtask_options_to_prompt(subtasks_and_options, prompt_style)
    return action_mask_prompt


FLAGS = flags.FLAGS

# Data
flags.DEFINE_string('traj_path', '../datasets/SGD/trajectories/Banks_1_trajectories.json', 'Path of demonstration trajectories')
#flags.DEFINE_string('graph_path', '../datasets/SGD/gt_graph/Banks_1_gt_graph.npy', 'Path to the subtask graph')
flags.DEFINE_string('graph_path', '../datasets/SGD/inferred_graph/SGD_Banks_1/inferred_graph_MaxRILP_purity0.9_depth5.npy', 'Path to the subtask graph')
flags.DEFINE_string('dataset', 'SGD', 'Name of dataset.')

# Hparam
flags.DEFINE_string('model', 'gpt-turbo', 'Name of LLM model.')
flags.DEFINE_float('temperature', 0, 'temperature for LLM sampling. if larger than 0, perform multi sampling')
flags.DEFINE_string('sampling', 'multi', 'How to sample actions')
flags.DEFINE_integer('num_shot', 5, 'number of demonstrations for few-shot prompting.')
flags.DEFINE_boolean('use_mask_prompt', False, 'whether to use action masking prompt')
flags.DEFINE_string('prompt_style', 'entire-concise', 'prompt style')
flags.DEFINE_multi_integer('seed', 1636423, 'seed for random demo prompt.')
flags.DEFINE_string('demo_method', 'seed', 'which demo selection method to use, seed or knn')

# Output
flags.DEFINE_string('output_dir', '', 'directory name to store prediction results')


def main(argv):
    assert len(FLAGS.output_dir) > 0, "Error: specify output directory name to store prediction results"
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    split_to_eval = 'test' if "WOZ" in FLAGS.traj_path else 'val'
    assert any([task_name in FLAGS.traj_path for task_name in TASK_SET_BY_DATASET[FLAGS.dataset.lower()]]), f"Error: uknown task for dataset {FLAGS.dataset}"
    
    if FLAGS.model == "gpt-turbo":
        from GPT.OpenAIAPIchat import getcompletion
    elif FLAGS.model == "gpt-3":
        from GPT.OpenAIAPI import getcompletion
    elif "codegen" in FLAGS.model:
        from GPT.codegen_completion import getcompletion
    elif "t5" in FLAGS.model:
        from GPT.t5_completion import getcompletion, sample_action_by_action

    # 0. data loading
    print(f'Loading trajectory @ {FLAGS.traj_path}')
    data_dict = json.load(open(FLAGS.traj_path))
    for i in data_dict.keys():
        task_name = i
        break
    if len(task_name.split('_')) > 1:
        short_task_name = f"{task_name.split('_')[0]}_{task_name.split('_')[1]}"
    else:
        short_task_name = task_name
    data_dict = data_dict[task_name]
    option_labels = data_dict['option_labels']
    subtask_labels = data_dict['subtask_labels']
    num_subtask = len(subtask_labels)
    
    if FLAGS.use_mask_prompt or (FLAGS.temperature > 0 and 'repeat' in FLAGS.sampling):
        assert short_task_name in FLAGS.graph_path, f"Error: loading graph of a wrong task. curr task={task_name}. graph path={FLAGS.graph_path}"
        gt_sop = load_gt_graph_sop(dataset_name = FLAGS.dataset, data_path = FLAGS.traj_path, graph_path=FLAGS.graph_path, task_name=task_name)
        neg_precond_path = FLAGS.graph_path[:FLAGS.graph_path.rfind('/')] + "/inferred_negative_precondition.json"
        neg_precond = json.load(open(neg_precond_path))
        precond = (gt_sop, neg_precond)
    else:
        precond = None

    # 1. Load demo prompt
    demo_path = FLAGS.traj_path.replace('trajectories.json', 'demonstrations.json')
    if '_' not in demo_path.split('/')[-1]:
        demo_path = demo_path.replace('.json','_demonstrations.json') # do this for MultiWOZ
    print(f'reading demo data @ {demo_path}')
    demo_dict_list = json.load(open(demo_path))
    print('Loading data... Done!')

    if FLAGS.demo_method == 'knn':
        demo_path = FLAGS.traj_path.replace('trajectories.json', 'demo_embeds.pt')
        import torch
        try:
            demo_embeds = torch.load(demo_path)
        except:
            d1 = torch.load(demo_path.replace('demo_embeds.pt','demo_embeds_pt1.pt'))
            d2 = torch.load(demo_path.replace('demo_embeds.pt','demo_embeds_pt2.pt'))
            d1.update(d2)
            demo_embeds = d1
    
    # 2. Predict action using model on validation set
    for seed in FLAGS.seed:
        t0 = time.time()
        # switch to new seed
        ### change output dump file name
        out_file_name = f"{FLAGS.output_dir}/DM_prediction_S{seed}.json"
        prompt_file_name = f"{FLAGS.output_dir}/prompt_S{seed}.json"
        FLAGS.append_flags_into_file(FLAGS.output_dir+f"/config_S{seed}")
        dd = FLAGS.flags_by_module_dict()
        flags_dict_list = [{v.name: v.value for v in vs} for k, vs in dd.items() if '.py' in k]
        flags_dict_list = [cand_dict for cand_dict in flags_dict_list if "dataset" in cand_dict]
        flags_dict = flags_dict_list[0]
        if isinstance(flags_dict, list):
            flags_dict = flags_dict[0]
        json.dump(flags_dict, open(FLAGS.output_dir+f"/config_S{seed}.json", 'w+'))
        
        ### switch to new demo with new seed
        demo_trajectories = None
        for demo_dict in demo_dict_list:
            if demo_dict['seed'] == seed and demo_dict['task_name'] == task_name:
                demo_trajectories = demo_dict['demo_trajectories']
        assert demo_trajectories is not None, "Error: could not find demo trajectory"
        ### Figure out max num demo
        max_num_shot = min(len(demo_trajectories), FLAGS.num_shot)
        failure_count = 0
        for num_demo_epi in range(max_num_shot,1,-1):
            try:
                max_num_demo_epi = num_demo_epi
                demo_prompt = get_demo_prompt(demo_trajectories[:num_demo_epi], precond, FLAGS.use_mask_prompt, subtask_labels, option_labels, FLAGS.prompt_style)
                output = getcompletion(demo_prompt, model=FLAGS.model, temperature=0) # turn off for max token test.
                break
            except Exception as e:
                failure_count += 1
                """
                if 'out of memory' in str(e):
                    print(f'Out of memory for {num_demo_epi} demos')
                else:
                    print('Unfortunate '+str(num_demo_epi))
                    print(e)"""
        if failure_count > 0:
            max_num_demo_epi = max_num_demo_epi - 1 # be safe
        print(f"Using {max_num_demo_epi} demo episodes with {len(demo_prompt)}-char")
    
        # Run experiment. Generate prompt and run LLM
        results, prompt_list = [], []
        record_num_demo_epi = []
        num_tokens_sum = 0
        eval_trajectories = [trajectory for trajectory in data_dict['trajectories'] if trajectory['split'] ==  split_to_eval]
        for trajectory in tqdm(eval_trajectories):
            subtasks_and_options = trajectory['subtasks_and_options'] # len() == number of options of SYSTEM and USER
            
            option_turn_count = 0
            response_turns, prompt_turns = [], []
            for turn_index, c in enumerate(subtasks_and_options):
                # At each system option turn, we ask model to predict the system option
                if FLAGS.dataset == 'SGD':
                    if c[0] != 'option' or c[1][0][0:6] != 'SYSTEM':
                        continue
                else:
                    if turn_index % 4 != 2:
                        continue

                if FLAGS.temperature > 0 and 'repeat' in FLAGS.sampling:
                    from util.logic_utils import predict_elig
                    completion = trajectory_to_completion(trajectory, num_subtask, end=turn_index)
                    elig_mask = np.stack([predict_elig(op_sop, completion) for op_sop in gt_sop], axis=1)[0]
                    elig_actions = [option for i, option in enumerate(option_labels) if elig_mask[i]]

                # perform knn demo selection
                if FLAGS.demo_method == 'knn':
                    name = trajectory['name']
                    embed = fetch_embed(name,option_turn_count,demo_embeds)
                    trajnames = find_closest_in_train(embed,demo_embeds,max_num_demo_epi)
                    #print(trajnames)
                    trajs = []
                    for traj in data_dict['trajectories']:
                        if traj['name'] in trajnames:
                            trajs.append(processdemo(traj))
                    demo_trajectories = trajs
                
                # 1. Generate prompt & run LLM to get response
                outputs = None
                num_demo_epi_used = 0
                for num_demo_epi in range(max_num_demo_epi,1,-1):
                    ### Get demo prompt. If exceeds token limit, reduce # demo
                    prefix = ""
                    if 'prefix' in FLAGS.prompt_style:
                        prefix = f"Your task is to act as a customer service system for {get_natural_task_name(task_name)}. "
                        prefix+= "Given a partial trajectory of dialog between user and system in terms of their actions and statuses, " 
                        prefix+= "your goal is to respond with the correct system action.\n\n"
                    
                    # Demo prompts
                    demo_prompt = get_demo_prompt(demo_trajectories[:num_demo_epi], precond, FLAGS.use_mask_prompt, subtask_labels, option_labels, FLAGS.prompt_style)
                    
                    # Task prompt
                    task_prompt = get_task_prompt(trajectory, turn_index, precond, FLAGS.use_mask_prompt, subtask_labels, option_labels, FLAGS.prompt_style)
                    
                    # 2. run LLM model for action prediction
                    prompt = prefix + demo_prompt + task_prompt
                    for _ in range(4):
                        try:
                            if FLAGS.temperature > 0 and 'repeat' in FLAGS.sampling:
                                outputs = sample_action_by_action(prompt, model=FLAGS.model, temperature=FLAGS.temperature, elig_actions=elig_actions, prompt_style=FLAGS.prompt_style)
                            else:
                                num_tokens = num_tokens_from_prompt(prompt) + 50
                                num_tokens_sum += num_tokens
                                outputs = getcompletion(prompt, model=FLAGS.model, temperature=FLAGS.temperature)
                            num_demo_epi_used = num_demo_epi
                            break
                        except Exception as e:
                            if 'out of memory' in str(e):
                                print(f'Out of memory for {num_demo_epi} demos')
                                break
                            elif 'rate limit' in str(e).lower():
                                print(e)
                                print(f'Rate limit @ {num_tokens_sum}.. sleeping 1min...')
                                num_tokens_sum = 0
                                time.sleep(60)
                                print(f'..done sleeping!')
                            else:
                                continue
                    if outputs is not None:
                        break
                # 3. Parse and dump result
                if outputs is None:
                    print('Error: LLM has no output. Usually due to rate limit.')
                    import ipdb; ipdb.set_trace()
                    assert False
                ###
                if isinstance(outputs, list): # multi sampling
                    options_list, subtasks_list = [], []
                    for output in outputs:
                        if 'concise' in FLAGS.prompt_style:
                            output = 'SYSTEM act: ' + output
                        elif 'symbolic' in FLAGS.prompt_style:
                            output = 'Actions: SYSTEM ' + output
                        else:
                            raise NotImplementedError
                        subtasks, options = prompt_to_subtask_options(output, FLAGS.prompt_style)
                        options_list.append(options)
                        subtasks_list.append(subtasks)
                    parsed_response = {'turn_id': option_turn_count,'USER': {'action':[],'status':[]},'SYSTEM': {'action': options_list, "status": subtasks_list}}
                    prompt = {'turn_id': option_turn_count, "prefix": prefix, "demo": demo_prompt, "task": task_prompt, "response": output, "action": options_list, "status": subtasks_list}
                else:
                    output = outputs
                    if 'concise' in FLAGS.prompt_style:
                        output = 'SYSTEM act: ' + output
                    elif 'symbolic' in FLAGS.prompt_style:
                        output = 'Actions: SYSTEM ' + output
                    else:
                        raise NotImplementedError
                    subtasks, options = prompt_to_subtask_options(output, FLAGS.prompt_style)
                    parsed_response = {'turn_id': option_turn_count,'USER': {'action':[],'status':[]},'SYSTEM': {'action': options, "status": subtasks}}
                    prompt = {'turn_id': option_turn_count, "prefix": prefix, "demo": demo_prompt, "task": task_prompt, "response": output, "action": options, "status": subtasks}
                    
                response_turns.append(parsed_response)
                prompt_turns.append(prompt)
                record_num_demo_epi.append(num_demo_epi_used)
                option_turn_count += 1
                
            t1 = time.time()
            record_for_eval = {'dialog_id': trajectory['name'], "turns": response_turns, "num_demo": sum(record_num_demo_epi)/len(record_num_demo_epi), 'cumulative running time': t1-t0}
            results.append(record_for_eval)
            
            """
            prompt_for_dump = {'dialog_id': trajectory['name'], "prompts": prompt_turns, "num_demo": sum(record_num_demo_epi)/len(record_num_demo_epi), 'cumulative running time': t1-t0}
            prompt_list.append(prompt_for_dump)"""
            
        f = open(out_file_name,'w+')
        json.dump(results,f,indent=2)
        print(f'Saved @ {out_file_name}')
        
        """
        f = open(prompt_file_name, 'w+')
        json.dump(prompt_list,f,indent=2)"""
            
        print(f'Experiment for {seed} is done!')
        
if __name__=='__main__':
    app.run(main)
