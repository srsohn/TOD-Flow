import copy
import random
import json
import numpy as np
from absl import flags,app

FLAGS = flags.FLAGS
flags.DEFINE_string('traj_path', '../datasets/SGD/trajectories/Banks_1_trajectories.json', 'Path of demonstration trajectories')
flags.DEFINE_string('task_name', '', 'name of the task')
flags.DEFINE_integer('num_shot', 20, 'number of demonstration')

SEED_SET = [51235435, 435232, 1636423, 745135]

def demo_trajectory_sampler(data_dict, task_name, num_shot):
    trajectories = data_dict[task_name]['trajectories']
    train_trajectories = [traj for traj in trajectories if traj['split']=='train']
    num_traj = len(train_trajectories)
    traj_indices = np.random.permutation(num_traj)[:num_shot].tolist()
    
    demo_trajectories = []
    for index in traj_indices:
        org_traj = train_trajectories[index]
        demo_traj = copy.deepcopy(org_traj)
        
        assert len(demo_traj['subtasks_and_options']) == len(demo_traj['subtask_and_option_indices'])
        subtasks_and_options = demo_traj['subtasks_and_options']
        
        # Sample random system option turn
        system_option_turn_indices = []
        for turn_index, turn_tuple in enumerate(subtasks_and_options):
            utterance_concat = "".join(turn_tuple[1])
            if turn_tuple[0]=='option' and 'system' in utterance_concat.lower():
                system_option_turn_indices.append(turn_index)
        rand_system_option_turn_index = random.choice(system_option_turn_indices)
        
        # Construct demo trajectory
        demo_traj['turn_index'] = rand_system_option_turn_index
        ### truncate until the randomly sampled turn index (i.e., history)
        demo_traj['entire_subtasks_and_options'] = org_traj['subtasks_and_options']
        demo_traj['entire_subtask_and_option_indices'] = org_traj['subtask_and_option_indices']
        #
        demo_traj['subtasks_and_options'] = demo_traj['subtasks_and_options'][:rand_system_option_turn_index]
        demo_traj['subtask_and_option_indices'] = demo_traj['subtask_and_option_indices'][:rand_system_option_turn_index]
        ### Add gt label
        demo_traj['gt_system_option'] = org_traj['subtasks_and_options'][rand_system_option_turn_index]
        demo_traj['gt_system_option_index'] = org_traj['subtask_and_option_indices'][rand_system_option_turn_index]
        
        demo_trajectories.append(demo_traj)
    
    return demo_trajectories
    
    
def main(argv):
    # 0. data loading
    data_dict = json.load(open(FLAGS.traj_path))
    if FLAGS.task_name == '':
        for i in data_dict.keys():
            task_name = i
            break
    else:
        task_name = FLAGS.task_name
    task_short_name = task_name[:task_name.find('_')+2]
    
    demo_list = []
    for seed in SEED_SET:
        random.seed(seed)
        np.random.seed(seed)
        demo_trajectories = demo_trajectory_sampler(data_dict, task_name, FLAGS.num_shot)
        
        # TODO put meta data and save
        demo_dict = dict(
            seed=seed,
            task_name=task_name,
            demo_trajectories=demo_trajectories,
        )
        demo_list.append(demo_dict)
    
    if 'trajectories.json' in FLAGS.traj_path:
        outfile = FLAGS.traj_path.replace('trajectories.json', 'demonstrations.json')
    else:
        outfile = FLAGS.traj_path.replace('.json', '_demonstrations.json')
    with open(outfile, "w") as f:
        json.dump(demo_list, f)
    print(f'saved data @ {outfile}')

if __name__=='__main__':
    app.run(main)