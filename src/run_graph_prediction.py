import sys
from util.graph_utils import GraphEvaluator, SubtaskGraph, get_gt_graph, expand_sop, OptionSubtaskGraphVisualizer, make_so_graph_dict, get_graph_sop
import numpy as np
from pathlib import Path
import math
import time
from util.data_utils import DataLoader
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'MultiWOZ', 'Name of dataset.')
flags.DEFINE_string('task', 'attraction', 'Name of task.')
flags.DEFINE_string('data_path', '../datasets/MultiWOZ/trajectories/Attraction_trajectories.json', 'path to train dataset file')
flags.DEFINE_string('output_dir', '../ilp_out', 'Directory to store logs.')
flags.DEFINE_boolean('visualize', False, 'Visualization flag')
flags.DEFINE_integer('verbose', 0, 'Verbosity level.')
flags.DEFINE_string('gt_graph_path', None, 'path to ground truth graph npy file. Necessary for outputting graph precision/recall metrics. E.g. ../datasets/SGD/gt_graph/Banks_1_gt_graph.npy')

# Algorithm-related
flags.DEFINE_enum('algo', 'CSILP', ['BCILP', 'SHDILP', 'CSILP', 'MSG2'], 'ILP Algorithm', )

flags.DEFINE_float('filter_threshold', 1, 'threshold top K fraction of trajectories according to the score')
flags.DEFINE_float('edge_purity_threshold', 0.95, 'threshold to determine the order between subtasks. range [0.5, 1.0]')
flags.DEFINE_boolean('traj_weigh', False, 'whether to apply trajectory weighting')

# Data
flags.DEFINE_float('discount_factor', 0.9, '')
flags.DEFINE_float('forward_discount', 0.9, '')
flags.DEFINE_float('backward_discount', 0.9, '')
flags.DEFINE_integer('negative_horizon', 4, '')
flags.DEFINE_float('positive_bias', 1.0, '')

# Beam
flags.DEFINE_integer('beam_width', 5, 'width of beam search')
flags.DEFINE_integer('beam_depth', 2, '')
flags.DEFINE_float('complexity_penalty', 0.0, '')
flags.DEFINE_float('min_score', 0.0, '')

# Decision tree (sklearn)
flags.DEFINE_integer('max_depth', 8, '')
flags.DEFINE_float('min_leaf_frac', 0.01, '')

# Input: ilp_data
# Output: graph

def initialize_ilp(task_name, subtask_labels, option_labels):
  if FLAGS.algo == 'SHDILP':
    from graph.should_ilp import SHDILP
    ilp_class = SHDILP
  elif FLAGS.algo == 'CSILP':
    from graph.cs_ilp import CSILP
    ilp_class = CSILP
  elif FLAGS.algo == 'BCILP':
    from graph.bc_ilp import BCILP
    ilp_class = BCILP
  elif FLAGS.algo == 'MSG2':
    from graph.ccao_ilp import MSG2
    ilp_class = MSG2
    
  # 3. Set up ILP arguments and hparam string
  ilp_args=dict()
  hparam_str = f"{FLAGS.algo}"
  if FLAGS.algo in ['MSG2']:
    hparam_str += f"_th={FLAGS.edge_purity_threshold}"
  elif FLAGS.algo in ['CSILP']:
    # advanced data processing
    ilp_args.update(dict(
      forward_discount=FLAGS.forward_discount,
      backward_discount=FLAGS.backward_discount,
      positive_bias=FLAGS.positive_bias,
    ))
    hparam_str += f"_fgam={FLAGS.forward_discount}_bgam={FLAGS.backward_discount}_pos={FLAGS.positive_bias}"
  elif FLAGS.algo in ['BCILP']:
    # simplified data processing
    pass
  
  if FLAGS.algo in ['SHDILP']:
    # Beam search
    ilp_args.update(dict(
      beam_width=FLAGS.beam_width,
      beam_depth=FLAGS.beam_depth,
      complexity_penalty=FLAGS.complexity_penalty,
      min_score=FLAGS.min_score,
    ))
    hparam_str += f"_bw={FLAGS.beam_width}_bd={FLAGS.beam_depth}_cp={FLAGS.complexity_penalty}_mins={FLAGS.min_score}"
  if FLAGS.algo in ['BCILP', 'CSILP']:
    # Use sklearn DT for ILP
    ilp_args.update(dict(
      max_depth=FLAGS.max_depth,
      min_leaf_frac=FLAGS.min_leaf_frac,
      max_features=0,
    ))
    hparam_str += f"_dep={FLAGS.max_depth}_leaf={FLAGS.min_leaf_frac}"
    
  #
  ilp = ilp_class(
    ilp_args,
    task_name=task_name,
    subtask_label=subtask_labels,
    option_label=option_labels
  )
  
  # 2. effect ilp
  from graph.effect_ilp import EffectILP
  effect_ilp_class = EffectILP
  effect_ilp = effect_ilp_class()
  
  return ilp, effect_ilp, hparam_str
  

def main(argv):
  assert FLAGS.edge_purity_threshold > 0.5 and FLAGS.edge_purity_threshold < 1.0, "edge_purity_threshold must be between [0.5 ~ 1.0]"
  SEPARATE_SUBTASK_OPTION = FLAGS.algo not in ['MSG2']
  # 1. load dataset
  dataset = DataLoader(
    dataset = FLAGS.dataset,
    task = FLAGS.task,
    data_path = FLAGS.data_path,
    split = 'train',
    dialog_flow_home_dir = "../"   # change this to where your dialog_flow_home_dir is relative to your wd!
  )
  so_ilp_data = dataset.to_so_ilp_data()
  ilp_data = dataset.to_ilp_data()
  task_name = list(so_ilp_data.keys())[0]
  if len(task_name.split('_')) > 1:
    short_task_name = f"{task_name.split('_')[0]}_{task_name.split('_')[1]}"
  else:
    short_task_name = task_name

  assert FLAGS.task.lower() == short_task_name.lower(), f"{FLAGS.task.lower()} != {short_task_name.lower()}"
  so_ilp_data = so_ilp_data[task_name]
  ilp_data = ilp_data[task_name]
  #
  if SEPARATE_SUBTASK_OPTION:
    subtask_labels = so_ilp_data['subtask_labels']
    option_labels = so_ilp_data['option_labels']
  else:
    subtask_labels = ilp_data['subtask_labels']
    option_labels = None
  
  # 2. Init ILP & output file name
  ilp, effect_ilp, hparam_str = initialize_ilp(task_name, subtask_labels, option_labels)
  output_filename = f"{FLAGS.output_dir}/{FLAGS.dataset}_{FLAGS.task}/inferred_graph_{hparam_str}"
  Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
  
  # 3. Infer effect & precondition
  print('Inferring effect...')
  effect_ilp.add_data(so_ilp_data)
  hard_effect_mat, soft_effect_mat = effect_ilp.infer_task()
  
  if SEPARATE_SUBTASK_OPTION:
    print('Inferring precondition...')
    ilp.add_data(so_ilp_data, soft_effect_mat)
  elif FLAGS.algo == 'MSG2':
    print('Inferring precondition...')
    order_data = dataset.to_order_data()
    order_data = order_data[task_name]
    ilp.add_data(ilp_data, order_data)
  
  if FLAGS.algo == 'MSG2':
    compact_inferred_sop_set, tind_by_layer, subtask_layer = ilp.infer_task(FLAGS.edge_purity_threshold, weighted=FLAGS.traj_weigh)
    inferred_sop_set = expand_sop(compact_inferred_sop_set, tind_by_layer, subtask_layer)
  else:
    inferred_sop_set = ilp.infer_task()
    tind_by_layer, subtask_layer = None, None
  
  # 5. Save graph. If needed, convert to option-subtask graph
  if FLAGS.algo in ['MSG2']:
    num_subtask = effect_ilp.num_subtask
    num_options = effect_ilp.num_option
    so_sop = [sop[:, :num_subtask] if sop is not None else None for sop in inferred_sop_set[num_subtask:]]
    so_sop = [sop if sop is None or (sop != 0).any() else None for sop in so_sop]
    so_sop = [sop[(sop != 0).any(1)] if sop is not None else None for sop in so_sop]
  else:
    so_sop = inferred_sop_set
  
  option_subtask_graph = make_so_graph_dict(
      so_ilp_data['subtask_labels'], # 45
      so_ilp_data['option_labels'], # 44
      so_sop, # 89
      soft_effect_mat,
  )
  np.save(output_filename, option_subtask_graph)
  print(f'Dump output @ {output_filename}')
  
  # 6. Visualize
  if FLAGS.visualize:
    print('Visualizing...')

    # 6.1 Save and visualize
    vis_filename = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/inferred_subtask_only_graph_{hparam_str}"
    so_vis_filename = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/inferred_graph_{hparam_str}"
    user_so_vis_filename = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/user_inferred_graph_{hparam_str}"
    system_so_vis_filename = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/system_inferred_graph_{hparam_str}"    
    #so_output_filename = f"{FLAGS.output_dir}/{FLAGS.dataset}_{FLAGS.task}/inferred_graph_{hparam_str}"
    vis_eff_filename = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/inferred_effect_{hparam_str}"

    Path(vis_filename).parent.mkdir(parents=True, exist_ok=True)
    print(f'visualize @ {so_vis_filename}')
    #ilp.visualize(vis_filename, inferred_sop_set, subtask_layer, show_index=True)
    effect_ilp.visualize(vis_eff_filename)
    visualizer = OptionSubtaskGraphVisualizer()
    dot = visualizer.visualize(so_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels)
    visualizer.render_and_save(dot, so_vis_filename)
    dot = visualizer.visualize(so_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels, hide_user=True)
    visualizer.render_and_save(dot, system_so_vis_filename)
    dot = visualizer.visualize(so_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels, hide_system=True)
    visualizer.render_and_save(dot, user_so_vis_filename)
    #g = SubtaskGraph(sop_by_subtask = inferred_sop_set, tind_by_layer=tind_by_layer)
    #np.save(output_filename, g)
  
  
  # 7. Evaluate against GT graph
  if FLAGS.gt_graph_path is not None:
    print('Evaluating against gt @{FLAGS.gt_graph_path} ...')
    gt = np.load(FLAGS.gt_graph_path, allow_pickle=True).item()# TODO make this an argument
    gt_sop = get_graph_sop(
        gt,
        subtask_list=so_ilp_data['subtask_labels'],
        option_list=so_ilp_data['option_labels'],
    )
    if FLAGS.visualize:
      gt_path = f"{FLAGS.output_dir}/visualize/{FLAGS.dataset}_{FLAGS.task}/gt_graph"
      visualizer = OptionSubtaskGraphVisualizer()
      dot = visualizer.visualize(gt_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels)
      visualizer.render_and_save(dot, gt_path)
      dot = visualizer.visualize(gt_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels, hide_user=True)
      visualizer.render_and_save(dot, gt_path+'_system')
      dot = visualizer.visualize(gt_sop, soft_effect_mat, effect_ilp.subtask_labels, effect_ilp.option_labels, hide_system=True)
      visualizer.render_and_save(dot, gt_path+'_user')
    is_system_option = np.array(['SYSTEM' in op for op in so_ilp_data['option_labels']], dtype=np.bool_)
    dump_path = f"{FLAGS.output_dir}/{FLAGS.dataset}_{FLAGS.task}"
    graph_eval = GraphEvaluator()
    precision, recall = graph_eval.eval_graph(so_sop, gt_sop, dump_path)
    print('Metrics vs. GT')
    print(f'precision: {precision.mean()*100:.2f}%')
    print(f'recall: {recall.mean()*100:.2f}%')
    print(f'system precision: {precision[is_system_option].mean()*100:.2f}%')
    print(f'system recall: {recall[is_system_option].mean()*100:.2f}%')
    print(f'user precision: {precision[~is_system_option].mean()*100:.2f}%')
    print(f'user recall: {recall[~is_system_option].mean()*100:.2f}%')

if __name__ == '__main__':
  # Environment loop flags.
  app.run(main)
