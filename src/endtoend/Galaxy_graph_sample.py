import os,sys
sys.path.append(os.getcwd())
import json
import numpy as np
from absl import app
from absl import flags
NUM_SAMPLES=7
from util.graph_utils import get_graph_sop


FLAGS = flags.FLAGS

flags.DEFINE_string('gt_path', '../outputs/standardized_trajectories_gt.json', 'Path of ground truth trajectories')
flags.DEFINE_string('pred_path', None, 'Path of predicted trajectories')
flags.DEFINE_string('out_path', None, 'Path of output json')
flags.DEFINE_string('agent', 'SYSTEM', 'Calculate metrics for <agent>')
flags.DEFINE_string('graph_mask', None, 'If not None, load a graph from path. Use graph to mask eligible actions using graph.')
flags.DEFINE_string('shd_graph_mask', None, 'If not None, load a graph from path. Use graph to eval should actions using graph.')
flags.DEFINE_string('neg_mask',None,'Negative mask, optional')
flags.DEFINE_string('can_factor','1','can factor, optional')
flags.DEFINE_string('shd_thresh','0','should threshold, optional')
flags.DEFINE_string('method','GALAXY','the base method we are improving upon')

# ======================== a bunch of helper functions for graph interactions ======================
def _map_set_vec(mapping, keys, vec):
  for key in keys:
    if key in mapping:
      vec[mapping[key]] = True

def _map_set_vec_2(mapping, keyss, vec):
  for i,keys in enumerate(keyss):
    for key in keys:
      if key in mapping:
        vec[i][mapping[key]] = True

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


def count_violations_should(eligs, ops):
  violated = eligs & ~ops
  return violated.sum(), eligs.sum()

def count_violations_2(eligs, ops):
  violated = ~eligs & ops
  return violated.sum(axis=1), ops.sum(axis=1)

# ======================== end of helper functions ======================

# validates that the trajectories and the predicted actions match in order
def validate_trajs(gt, pred):
  # validate
  id_to_gt = {traj['dialog_id']: traj for traj in gt}
  id_to_pred = {traj['dialog_id']: traj for traj in pred}
  all_acts = set()
  all_statuses = set()
  messed = []
  for traj in pred:
    assert traj['dialog_id'] in id_to_gt, 'Predicted trajectory must be in ground truth trajectories'
    gt_traj = id_to_gt[traj['dialog_id']]
    try:
      assert len(traj['turns']) == len(gt_traj['turns'])
    except:
      print(traj['dialog_id'])
      print(len(traj['turns']))
      print(len(gt_traj['turns']))
      messed.append(traj['dialog_id'])
    for turn, gt_turn in zip(traj['turns'], gt_traj['turns']):
      for agent in ['USER', 'SYSTEM']:
        if agent == 'USER' and len(gt_turn[agent]) == 0:
          continue# edge case in specs where system has 2 turns in a row
        all_acts.update(gt_turn[agent]['action'])
        all_statuses.update(gt_turn[agent]['status'])

  return list(sorted(all_acts)), list(sorted(all_statuses)),messed

# the main selection function
def dact_traj_multi_sample_metrics_report(gt, pred, graph_sop=None, neg_precond_mat=None, should_sop=None, messed = []):
  """
  gt: ground truth trajectory (used for looking up ground truth dialog history actions and statuses)
  pred: predicted next actions for each candidate response
  graph_sop: the can and shouldn't graph
  neg_precond_map: negative precondition graph, not used for this experiment
  should_sop: the shd graph
  messed: which dialogs failed to validate and therefore we should not try to do graph filter on it
  """
  
  multfactor = int(FLAGS.can_factor)
  shdfactor = 1
  shdthresh = int(FLAGS.shd_thresh)
  # validate
  id_to_gt = {traj['dialog_id']: traj for traj in gt}
  id_to_pred = {traj['dialog_id']: traj for traj in pred}
  all_acts, all_statuses, messed = validate_trajs(gt, pred)

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
    if traj['dialog_id'] in messed:
      continue
    gt_traj = id_to_gt[traj['dialog_id']]
    _cur_x_gt = np.zeros(len(all_statuses), dtype=np.bool_)
    for turn, gt_turn in zip(traj['turns'], gt_traj['turns']):
      for agent in ['USER','SYSTEM']:
        if agent == 'USER' and len(gt_turn[agent]) == 0:
          continue# edge case in specs where system has 2 turns in a row
        # dialogue acts
        _y_gt = np.zeros(len(dialogue_acts), dtype=np.bool_)
        _y_pred = np.zeros((10,len(dialogue_acts)), dtype=np.bool_)
        _map_set_vec(act_to_id, gt_turn[agent]['action'], _y_gt)
        _map_set_vec_2(act_to_id, turn[agent]['action'], _y_pred)
        Y_gt.append(_y_gt)
        Y_pred.append(_y_pred)
        Y_agent.append(agent)

        # statuses
        _x_gt = np.zeros(len(all_statuses), dtype=np.bool_)
        _map_set_vec(status_to_id, gt_turn[agent]['status'], _x_gt)
        X_gt.append(_cur_x_gt)# use the previous status for current turn
        _cur_x_gt = _cur_x_gt | _x_gt

  Y_gt = np.stack(Y_gt)
  Y_pred = np.stack(Y_pred)
  Y_agent = np.array(Y_agent)
  X_gt = np.stack(X_gt)

  masked_labels = np.arange(is_agent_act.sum())
  masked_Y_gt = Y_gt[Y_agent == FLAGS.agent][:, is_agent_act] * masked_labels
  masked_Y_pred = Y_pred[Y_agent == FLAGS.agent][:,:, is_agent_act] * masked_labels
  
  if 'star' in FLAGS.method.lower():
    ord = [5,4,3,2,1]
  else:
    ord = [4,3,2,1,0]
  

  if graph_sop is not None:
    aug_graph_sop = [True] + graph_sop# include omit action
    elig_mask = np.stack([predict_elig(op_sop, X_gt) for op_sop in aug_graph_sop], axis=1)

    if neg_precond_mat is not None: # (29, 29)
      elig_mask_from_neg_precond = ~ (neg_precond_mat @ X_gt.T) # (#option, 354)
      aug_elig_mask_from_neg_precond = np.append(np.ones((1, X_gt.shape[0]), dtype=bool), elig_mask_from_neg_precond, 0) # (#option+1, 354)
      aug_elig_mask_from_neg_precond = aug_elig_mask_from_neg_precond.T
      elig_mask = elig_mask * aug_elig_mask_from_neg_precond


    if should_sop is not None:
      aug_should_sop = [False] + should_sop# include omit action
      should_mask = np.stack([predict_elig(shd_sop, X_gt) for shd_sop in aug_should_sop], axis=1)
    else:
      should_mask = False

    masked_elig = elig_mask[Y_agent == FLAGS.agent][:, is_agent_act]
    if isinstance(should_mask, np.ndarray):
      masked_shld = should_mask[Y_agent == FLAGS.agent][:, is_agent_act]
    else:
      masked_shld = None
    violations, total_exec = count_violations(masked_elig, masked_Y_gt > 0)
    print(f'{violations}/{total_exec} violations between graph and ground truth DM actions')
    
    # This is the part where we calculate total violations of can, shdnt and shd for each candidate
    scores = np.zeros((len(masked_Y_gt), NUM_SAMPLES))
    for j in range(len(masked_Y_gt)):
      for i in range(NUM_SAMPLES):
        violations, total_exec = count_violations(masked_elig[j:j+1], masked_Y_pred[j:j+1,i,:] > 0)
        if masked_shld is not None:
          vv, tt = count_violations_should(masked_shld[j:j+1], masked_Y_pred[j:j+1,i,:] > 0)
          vv -= shdthresh
          if vv < 0:
              vv = 0
          violations *= multfactor
          total_exec *= multfactor
          violations += vv*shdfactor
          total_exec += tt*shdfactor
        scores[j,i] = violations/(total_exec+0.00001)
    scorechoices = np.zeros(len(masked_Y_gt)).tolist()
    # and here is where we pick the candidates with least violations for each response
    for i in range(len(masked_Y_gt)):
      minscore = 10000
      for j in ord:
        if j in maskoutlist:
          continue
        if scores[i,j] <= minscore:
          scorechoices[i] = j
          minscore = scores[i,j]

  return scorechoices

def main(argv):
  with open(FLAGS.gt_path) as f:
    gt = json.load(f)
  with open(FLAGS.pred_path) as f:
    pred = json.load(f)
  namedict = {'galaxy':'../datasets/MultiWOZ/e2e/galaxy_7full_pred.json','galaxystar':'../datasets/MultiWOZ/e2e/galaxy_7full_pred.json','hdno':'../datasets/MultiWOZ/e2e/hdno_7_pred.json','hdsa':'../datasets/MultiWOZ/e2e/hdsa_7new_pred.json'}
  fpath = namedict[FLAGS.method.lower()]
  with open(fpath) as f:
    five = json.load(f)
  neg_precond_mat=None
  if FLAGS.graph_mask is not None:
    gt_raw = np.load(FLAGS.graph_mask, allow_pickle=True).item()
    all_acts, all_statuses,messed = validate_trajs(gt, pred)
    gt_sop = get_graph_sop(
        gt_raw,
        subtask_list=all_statuses,
        option_list=all_acts,
    )
    neg_precond = FLAGS.neg_mask
    
    if neg_precond is not None:
      neg_precond = json.load(open(neg_precond))
      assert len(set(all_acts) - set(neg_precond['option_labels'])) == 0
      assert len(set(all_statuses) - set(neg_precond['subtask_labels'])) == 0
      op_inv_index = [neg_precond['option_labels'].index(s) for s in all_acts]
      su_inv_index = [neg_precond['subtask_labels'].index(s) for s in all_statuses]
      neg_precond_mat = np.stack(neg_precond['precondition_vectors'])
      neg_precond_mat = neg_precond_mat[op_inv_index, :]
      neg_precond_mat = neg_precond_mat[:, su_inv_index]
    if FLAGS.shd_graph_mask is not None:
      sgt_raw = np.load(FLAGS.shd_graph_mask, allow_pickle=True).item()
      sgt_sop = get_graph_sop(
        sgt_raw,
        subtask_list=all_statuses,
        option_list=all_acts,
        empty_value=False
      )
    else:
      sgt_sop = None

  else:
    gt_sop = None
    sgt_sop = None
  

  choices = dact_traj_multi_sample_metrics_report(gt, pred, graph_sop = gt_sop, neg_precond_mat=neg_precond_mat, should_sop=sgt_sop, messed = messed)
  print(choices)
  count = 0
  nfive = {}
  for d in pred:
    name = d['dialog_id']
    if name in messed:
      continue
    sname = name.lower()[:-5]
    for i in range(len(d['turns'])):
      five[sname][i] = {"response":five[sname][i][choices[count]]}
      count += 1
    nfive[sname] = five[sname]
  assert count == len(choices)
  json.dump(nfive,open(FLAGS.out_path,'w+'),indent=2)

maskoutlist = []

if __name__ == '__main__':
  app.run(main)
