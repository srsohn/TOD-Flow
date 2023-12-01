from GPT.GPTutils import NUM_SAMPLES
import numpy as np
import json
from sklearn.metrics import classification_report
import pandas as pd

AGENT = 'SYSTEM'

def normalize_txt(text):
  text = text.strip() # remove leading/trailing white space
  text = text.lower()
  text = text.replace('_', ' ')
  text = text.replace("[\'", '')
  text = text.replace("\']", '')
  return text

# https://stackoverflow.com/questions/2460177/edit-distance-in-python
def edit_distance(s1, s2):
  s1 = normalize_txt(s1)
  s2 = normalize_txt(s2)
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2+1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    distances = distances_
  return distances[-1]

def _map_set_vec(mapping, keys, vec, approx=False, approx_thresh=0.3):
  for key in keys:
    if key in mapping:
      vec[mapping[key]] = True
    elif approx:
      valid_keys = list(mapping.keys())
      dists = [edit_distance(vk, key) for vk in valid_keys]
      closest_i = np.argmin(dists)
      min_dist = dists[closest_i]
      if (min_dist / len(key)) < approx_thresh:
        vec[mapping[valid_keys[closest_i]]] = True

def _map_set_vec_2(mapping, keys_list, vec, approx=False, approx_thresh=0.3):
  for i,keys in enumerate(keys_list):
    for key in keys:
      if key in mapping:
        vec[i][mapping[key]] = True
      elif approx:
        valid_keys = list(mapping.keys())
        dists = [edit_distance(vk, key) for vk in valid_keys]
        closest_i = np.argmin(dists)
        min_dist = dists[closest_i]
        if (min_dist / len(key)) < approx_thresh:
          vec[i][mapping[valid_keys[closest_i]]] = True

def predict_elig(sop, comp):
  if sop is None:
    return np.ones(comp.shape[0], dtype=np.bool_)
  if isinstance(sop, bool):
    if sop:
      return np.ones(comp.shape[0], dtype=np.bool_)
    else:
      return np.zeros(comp.shape[0], dtype=np.bool_)
  threshs = np.abs(sop).sum(1)
  comp_pn = comp * 2 - 1
  preds = ((comp_pn @ sop.T) >= threshs).any(1)
  return preds

def count_violations(eligs, ops):
  violated = ~eligs & ops
  return violated.sum(), ops.sum()

def count_violations_2(eligs, ops):
  violated = ~eligs & ops
  return violated.sum(axis=1), ops.sum(axis=1)

def validate_trajs(gt, pred):
  # validate
  id_to_gt = {traj['dialog_id']: traj for traj in gt}
  id_to_pred = {traj['dialog_id']: traj for traj in pred}
  all_acts = set()
  all_statuses = set()
  for traj in pred:
    assert traj['dialog_id'] in id_to_gt, f"Dialog id={traj['dialog_id']} does not exist in GT validation labels"
    gt_traj = id_to_gt[traj['dialog_id']]
    assert len(traj['turns']) == len(gt_traj['turns']), f"dialog_id={traj['dialog_id']}: {len(traj['turns'])} != {len(gt_traj['turns'])}"
    for turn, gt_turn in zip(traj['turns'], gt_traj['turns']):
      for agent in ['USER', 'SYSTEM']:
        if agent == 'USER' and len(gt_turn[agent]) == 0:
          # assert len(turn[agent]) == 0
          continue# edge case in specs where system has 2 turns in a row
        #assert set(turn[agent]['status']) == set(gt_turn[agent]['status'])
        all_acts.update(gt_turn[agent]['action'])
        all_statuses.update(gt_turn[agent]['status'])

  return list(sorted(all_acts)), list(sorted(all_statuses))

def dact_traj_metrics_report(masked_Y_gt, masked_Y_pred_, Y_agent, X_gt, is_agent_act, graph_sop=None, neg_precond_mat=None, should_sops=None, verbose=True):
  '''
  Print metrics report for dialog act trajectories
  '''
  # masked_Y_pred: [num_steps x num_options]
  if not isinstance(graph_sop, list):
    graph_sops = [graph_sop]
    neg_precond_mats = [neg_precond_mat]
  else:
    graph_sops = graph_sop
    neg_precond_mats = neg_precond_mat
    
  report_dict_list = []
  for graph_sop, neg_precond_mat, should_sop in zip(graph_sops, neg_precond_mats, should_sops):
    masked_Y_pred = masked_Y_pred_.copy()
    if graph_sop is not None or neg_precond_mat is not None:
      if graph_sop is not None:
        aug_graph_sop = [True] + graph_sop# include omit action
        elig_mask = np.stack([predict_elig(op_sop, X_gt) for op_sop in aug_graph_sop], axis=1)
      else:
        elig_mask = 1
        
      if neg_precond_mat is not None: # (29, 29)
        elig_mask_from_neg_precond = ~ (neg_precond_mat @ X_gt.T) # (#option, 354)
        aug_elig_mask_from_neg_precond = np.append(np.ones((1, X_gt.shape[0]), dtype=bool), elig_mask_from_neg_precond, 0) # (#option+1, 354)
        aug_elig_mask_from_neg_precond = aug_elig_mask_from_neg_precond.T
        elig_mask = elig_mask * aug_elig_mask_from_neg_precond
        
      masked_elig = elig_mask[Y_agent == AGENT][:, is_agent_act]
      if verbose:
        violations, total_exec = count_violations(masked_elig, masked_Y_gt > 0)
        print(f'{violations}/{total_exec} violations between graph and ground truth DM actions')
      if verbose:
        violations, total_exec = count_violations(masked_elig, masked_Y_pred > 0)
        print(f'{violations}/{total_exec} violations between graph and predicted DM actions')

      masked_Y_pred = masked_Y_pred * masked_elig
    
    if should_sop is not None:
      aug_should_sop = [False] + should_sop# include omit action
      #print('========')
      #print(X_gt)
      should_mask = np.stack([predict_elig(shd_sop, X_gt) for shd_sop in aug_should_sop], axis=1)
      masked_shld = should_mask[Y_agent == AGENT][:, is_agent_act]
      masked_Y_predicted = np.logical_or((masked_Y_pred > 0), masked_shld)
      #print(masked_shld)
    else:
      masked_Y_predicted = masked_Y_pred > 0
    report_dict2 = classification_report(
        (masked_Y_gt > 0).flatten(), masked_Y_predicted.flatten(),
        target_names=['Omitted', 'Predicted'],
        zero_division=0,
        digits=3,
        output_dict=True
    )
    report_dict2['post'] = 'None'
    if verbose:
      print(pd.DataFrame(report_dict2).transpose().to_string())
    report_dict_list.append(report_dict2)
  return report_dict_list

def standardize_dact(gt, pred, near_option_mapping, multi):
  # validate
  id_to_gt = {traj['dialog_id']: traj for traj in gt}
  id_to_pred = {traj['dialog_id']: traj for traj in pred}
  all_acts, all_statuses = validate_trajs(gt, pred)

  dialogue_acts = np.array(['<omit>'] + all_acts)
  is_agent_act = np.array(['SYSTEM' in a for a in dialogue_acts])
  is_agent_act[0] = True
  act_to_id = {a: i for i, a in enumerate(dialogue_acts)}
  status_to_id = {a: i for i, a in enumerate(all_statuses)}

  Y_gt = []
  Y_pred = []
  Y_agent = []
  X_gt = []
  for traj in pred:
    gt_traj = id_to_gt[traj['dialog_id']]
    _cur_x_gt = np.zeros(len(all_statuses), dtype=np.bool_)
    for turn, gt_turn in zip(traj['turns'], gt_traj['turns']):
      for agent in ['USER','SYSTEM']:
        if agent == 'USER' and len(gt_turn[agent]) == 0:
          continue# edge case in specs where system has 2 turns in a row
        # dialogue acts
        _y_gt = np.zeros(len(dialogue_acts), dtype=np.bool_)
        if multi:
          num_samples = len(turn['SYSTEM']['action']) # user is not multi sampled. so use system's action dim.
          _y_pred = np.zeros((num_samples, len(dialogue_acts)), dtype=np.bool_)
        else:
          _y_pred = np.zeros((len(dialogue_acts)), dtype=np.bool_)
        _map_set_vec(act_to_id, gt_turn[agent]['action'], _y_gt)
        if multi:
          _map_set_vec_2(act_to_id, turn[agent]['action'], _y_pred, approx=near_option_mapping)
        else:
          _map_set_vec(act_to_id, turn[agent]['action'], _y_pred, approx=near_option_mapping)
        Y_gt.append(_y_gt)
        Y_pred.append(_y_pred)
        Y_agent.append(agent)

        # statuses
        _x_gt = np.zeros(len(all_statuses), dtype=np.bool_)
        _map_set_vec(status_to_id, gt_turn[agent]['status'], _x_gt)
        X_gt.append(_cur_x_gt)# use the previous status for current turn
        _cur_x_gt = _cur_x_gt | _x_gt

  Y_gt = np.stack(Y_gt) # [num_total_turns, action_dim]
  Y_pred = np.stack(Y_pred)
  Y_agent = np.array(Y_agent)
  X_gt = np.stack(X_gt)
  
  masked_labels = np.arange(is_agent_act.sum())
  #             Y_gt[SYSTEM turn only][:, SYSTEM action only]
  masked_Y_gt = Y_gt[Y_agent == AGENT][:, is_agent_act] * masked_labels
  if multi:
    masked_Y_pred_ = Y_pred[Y_agent == AGENT][:,:, is_agent_act] * masked_labels
  else:
    masked_Y_pred_ = Y_pred[Y_agent == AGENT][:,is_agent_act] * masked_labels
  return masked_Y_gt, masked_Y_pred_, Y_agent, X_gt, is_agent_act, all_acts, all_statuses

def dact_traj_multi_sample_metrics_report(masked_Y_gt, masked_Y_pred_, Y_agent, X_gt, is_agent_act, graph_sop=None, neg_precond_mat=None, should_sops=None, verbose=False):
  '''
  Print metrics report for dialog act trajectories
  '''
  #Y_gt, Y_pred, Y_agent, X_gt, is_agent_act = standardize_dact(gt, pred, near_option_mapping)
  # masked_Y_pred: [num_steps x num_samples x num_options]
  assert graph_sop is not None, "No graph given. But, multi sampling eval requires graph!"
  if not isinstance(graph_sop, list):
    graph_sops = [graph_sop]
    neg_precond_mats = [neg_precond_mat]
  else:
    graph_sops = graph_sop
    neg_precond_mats = neg_precond_mat
  
  report_dict_list = []
  for graph_sop, neg_precond_mat, should_sop in zip(graph_sops, neg_precond_mats, should_sops):
    masked_Y_pred = masked_Y_pred_.copy()
    if graph_sop is not None:
      aug_graph_sop = [True] + graph_sop# include omit action
      elig_mask = np.stack([predict_elig(op_sop, X_gt) for op_sop in aug_graph_sop], axis=1)
    else:
      elig_mask = True
    if should_sop is not None:
      aug_should_sop = [False] + should_sop# include omit action
      should_mask = np.stack([predict_elig(shd_sop, X_gt) for shd_sop in aug_should_sop], axis=1)
    else:
      should_mask = False
    if neg_precond_mat is not None: # (29, 29)
      elig_mask_from_neg_precond = ~ (neg_precond_mat @ X_gt.T) # (#option, 354)
      aug_elig_mask_from_neg_precond = np.append(np.ones((1, X_gt.shape[0]), dtype=bool), elig_mask_from_neg_precond, 0) # (#option+1, 354)
      aug_elig_mask_from_neg_precond = aug_elig_mask_from_neg_precond.T
      elig_mask = elig_mask * aug_elig_mask_from_neg_precond
    
    # extract only agent-related entries
    if isinstance(elig_mask, np.ndarray):
      masked_elig = elig_mask[Y_agent == AGENT][:, is_agent_act]
    else:
      masked_elig = elig_mask
    if isinstance(should_mask, np.ndarray):
      masked_shld = should_mask[Y_agent == AGENT][:, is_agent_act]
    else:
      masked_shld = should_mask
    #print(masked_shld)
    """
    violations, total_exec = count_violations(masked_elig, masked_Y_gt > 0)
    if verbose:
      print(f'{violations}/{total_exec} violations between graph and ground truth DM actions')"""

    masked_Y_predicted = masked_Y_pred > 0
    # masked_Y_predicted: [num_steps x num_samples x num_options]
    scores = np.zeros((len(masked_Y_gt), NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
      violations, total_exec = count_violations(masked_elig, masked_Y_predicted[:,i,:])
      scores[:,i] = violations/(total_exec+0.00001)
      if verbose:
        print(f'{scores} violations between graph and predicted DM actions')
      masked_Y_predicted[:,i,:] = np.logical_or(masked_Y_predicted[:,i,:], masked_shld)
      masked_Y_predicted[:,i,:] = masked_Y_predicted[:,i,:] * masked_elig # elem-wise multi
      
    # masked_Y_gt: [num_steps x num_options]
    maxchoices = np.zeros((len(masked_Y_gt),len(masked_Y_gt[0])))
    scorechoices = np.zeros((len(masked_Y_gt),len(masked_Y_gt[0])))
    majorchoices = np.zeros((len(masked_Y_gt),len(masked_Y_gt[0])))
    for i in range(len(masked_Y_gt)):
      majors = {}
      counts = np.count_nonzero(masked_Y_predicted[i], axis=1) # [num_samples]
      maxcount = counts.max()
      minscore = scores[i].min()
      for j in range(NUM_SAMPLES-1, -1, -1):
        s = binarytostr(masked_Y_predicted[i,j])
        if s not in majors:
          majors[s] = []
        majors[s] += [j]
        if counts[j] == maxcount:
          maxchoices[i] = masked_Y_predicted[i,j]
        if scores[i,j] == minscore:
          scorechoices[i] = masked_Y_predicted[i,j]
      maxmajor = 0
      for name in majors:
        if len(majors[name]) > maxmajor:
          maxmajor = len(majors[name])
          majorchoices[i] = masked_Y_predicted[i,majors[name][0]]

    masked_Y_gt_copied = np.zeros((masked_Y_gt.shape[0],NUM_SAMPLES,masked_Y_gt.shape[1]))
    for i in range(NUM_SAMPLES):
      masked_Y_gt_copied[:,i,:] = masked_Y_gt
    masked_Y_gt_copied = masked_Y_gt_copied.reshape(masked_Y_gt_copied.shape[0],-1)
    masked_Y_predicted = masked_Y_predicted.reshape(masked_Y_predicted.shape[0],-1)
    report_dict_uniform = classification_report(
        (masked_Y_gt_copied > 0).flatten(), masked_Y_predicted.flatten(),
        target_names=['Omitted', 'Predicted'],
        zero_division=0,
        digits=3,
        output_dict=True
    )
    report_dict_uniform['post'] = 'uniform'

    masked_Y_predicted = majorchoices
    report_dict_major = classification_report(
        (masked_Y_gt > 0).flatten(), masked_Y_predicted.flatten(),
        target_names=['Omitted', 'Predicted'],
        zero_division=0,
        digits=3,
        output_dict=True
    )
    report_dict_major['post'] = 'major'
    #report_dict_list.append(report_dict_major)
    
    masked_Y_predicted = scorechoices
    report_dict_score = classification_report(
        (masked_Y_gt > 0).flatten(), masked_Y_predicted.flatten(),
        target_names=['Omitted', 'Predicted'],
        zero_division=0,
        digits=3,
        output_dict=True
    )
    report_dict_score['post'] = 'violation'
    #report_dict_list.append(report_dict_score)

    masked_Y_predicted = maxchoices
    report_dict_max = classification_report(
        (masked_Y_gt > 0).flatten(), masked_Y_predicted.flatten(),
        target_names=['Omitted', 'Predicted'],
        zero_division=0,
        digits=3,
        output_dict=True
    )
    report_dict_max['post'] = 'max'


    report_dict_list.append( (report_dict_max, report_dict_score, report_dict_major, report_dict_uniform) )
  return report_dict_list

def binarytostr(li):
  out = ''
  for i in li:
    out += str(i)
  return out