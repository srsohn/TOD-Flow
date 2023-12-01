from logging import raiseExceptions
from typing import Optional, List

import os
import numpy as np
import itertools, math
from util import logic_utils
from util.data_utils import DataLoader
from dataclasses import dataclass
from IPython import embed

def load_gt_graph_sop(dataset_name, data_path, graph_path, task_name):
    # 1. load GT subtask graph (sop format)
    dataset = DataLoader(
        dataset = dataset_name,
        task = task_name,
        data_path = data_path,
        split = 'train',
        dialog_flow_home_dir = "../"   # change this to where your dialog_flow_home_dir is relative to your wd!
    )
    so_ilp_data = dataset.to_so_ilp_data()
    #
    print(f'reading data @ {graph_path}')
    gt_raw = np.load(graph_path, allow_pickle=True).item()
    gt_sop = get_graph_sop(
        gt_raw,
        subtask_list=so_ilp_data[task_name]['subtask_labels'],
        option_list=so_ilp_data[task_name]['option_labels'],
    )
    return gt_sop
  
@dataclass
class SubtaskGraph:
  W_a: np.ndarray
  W_o: np.ndarray
  ORmat: np.ndarray
  ANDmat: np.ndarray
  tind_by_layer: list
  sop_by_subtask: list

  def __init__(
      self,
      subtask_label: Optional[list] = None,
      W_a: Optional[np.ndarray] = None,
      W_o: Optional[np.ndarray] = None, # For GRProp
      ANDmat: Optional[np.ndarray] = None,
      ORmat: Optional[np.ndarray] = None, # For eligibility prediction
      sop_by_subtask: Optional[list] = None, # For precision-recall
      tind_by_layer: Optional[list] = None,
  ):
    self.subtask_label = subtask_label
    self.tind_by_layer = tind_by_layer
    if W_a is not None and W_o is not None:
      self._initialize_from_weight(W_a, W_o)
    elif ORmat is not None and ANDmat is not None:
      self._initialize_from_matrix(ANDmat, ORmat)
    elif sop_by_subtask is not None:
      self._initialize_from_sop(sop_by_subtask)
    else:
      assert False, 'no input to initialize from'

  def _initialize_from_weight(self, W_a, W_o):
    num_or = W_a[0].shape[1]
    num_and = 0
    for wo in W_o:
      num_or = num_or + wo.shape[0]
      num_and = num_and + wo.shape[1]
    self.num_or = num_or
    self.num_and = num_and
    #
    ANDmat, ORmat = self.weight_to_matrix(W_a, W_o, num_and, num_or)
    self.ANDmat = ANDmat
    self.ORmat = ORmat
    #
    self.sop_by_subtask = self.matrix_to_sop(ANDmat, ORmat)

  def _initialize_from_matrix(self, ANDmat, ORmat):
    self.num_or, self.num_and = ORmat.shape
    #
    W_a, W_o, tind_by_layer = self.matrix_to_weight(ANDmat, ORmat)
    self.W_a = W_a
    self.W_o = W_o
    self.tind_by_layer = tind_by_layer
    self.sop_by_subtask = self.matrix_to_sop(ANDmat, ORmat)

  def _initialize_from_sop(self, sop_by_subtask):
    num_subtask = len(sop_by_subtask)
    for sop in sop_by_subtask:
      if sop is not None:
        assert sop.shape[1] == num_subtask
    # to ANDmat, ORmat
    ANDmat, ORmat = self.sop_to_matrix(sop_by_subtask)
    self.ANDmat = ANDmat
    self.ORmat = ORmat
    #
    W_a, W_o, tind_by_layer = self.matrix_to_weight(ANDmat, ORmat)
    self.W_a = W_a
    self.W_o = W_o
    self.tind_by_layer = tind_by_layer

  # =====================
  def matrix_to_weight(self, ANDmat, ORmat):
    return None, None, None

  def weight_to_matrix(self, W_a, W_o, num_and, num_or):
    # ANDmat <-> W_a
    num_layers = len(W_a)
    num_and_by_layer = [wa.shape[0] for wa in W_a]
    bias=0
    and_layer_idx = []
    for nand in num_and_by_layer:
      and_layer_idx.append([*range(bias, bias+nand)])
      bias = bias + nand
    #
    or_index_bias = 0
    and_index_bias = 0
    recon_andmat = np.zeros((num_and, num_or), dtype=bool)
    recon_ormat = np.zeros_like((num_or, num_and), dtype=bool)
    cumulative_or_indices = []
    for i in range(num_layers):
      num_and_nodes_in_current_layer = len(and_layer_idx[i])
      # OR -> AND (ANDmat & W_a)
      wa = W_a[i]
      cumulative_or_indices = cumulative_or_indices + tind_by_layer[i]
      next_and_index_bias = and_index_bias + num_and_nodes_in_current_layer
      #
      recon_andmat[and_index_bias:next_and_index_bias, current_or_indices] = wa

      # AND -> OR (ORmat & W_o)
      wo = W_o[i]
      current_or_indices = tind_by_layer[i+1]
      recon_ormat[current_or_indices, and_index_bias:next_and_index_bias] = wo
      and_index_bias = next_and_index_bias
    return andmat, ormat

  def matrix_to_sop(self, ANDmat, ORmat):
    sop_by_subtask = []
    for subtask_index, or_vec in enumerate(ORmat):
      andvec_list = []
      andnode_indices = or_vec.nonzero()[0]
      if len(andnode_indices) == 0: # first layer
        sop_by_subtask.append(True) # no precondition
      else:
        #if len(andnode_indices) > 1:
        #  import ipdb; ipdb.set_trace()
        for andnode_index in andnode_indices:
          andvec = ANDmat[andnode_index]
          andvec_list.append(andvec)
        sop_mat = np.stack(andvec_list).astype(int)
        sop_by_subtask.append(sop_mat)
    return sop_by_subtask

  def sop_to_matrix(self, sop_by_subtask):
    num_subtask = len(sop_by_subtask)
    # populate AND nodes & fillout ANDmat
    hash_list = []
    andvec_list = []
    for subtask_index, sop in enumerate(sop_by_subtask):
      if sop is None:
        continue
      code_list = logic_utils.batch_bin_encode(sop)
      for ind, code in enumerate(code_list):
        if code not in hash_list:
          hash_list.append(code)
          andvec = sop[ind]
          andvec_list.append(andvec)
    if len(andvec_list) == 0: # All the subtask has NO precondition
      ANDmat = np.zeros((0,num_subtask))
    else:
      ANDmat = np.stack(andvec_list)
    num_and = len(andvec_list)

    # fillout ORmat
    ORmat = np.zeros((num_subtask, num_and), dtype=bool)
    for subtask_index, sop in enumerate(sop_by_subtask):
      if sop is None:
        continue
      code_list = logic_utils.batch_bin_encode(sop)
      for ind, code in enumerate(code_list):
        assert code in hash_list
        and_ind = hash_list.index(code)
        ORmat[subtask_index, and_ind] = 1
    return ANDmat, ORmat

  def predict_next_subtask(self, completion_matrix):
    num_data, num_subtask = completion_matrix.shape
    eligibility = self.compute_eligibility(completion_matrix)
    eligibility_trace = np.ones((num_subtask))
    ever_elig = np.zeros((num_subtask), dtype=bool)
    _lambda = 0.9
    probs = []
    for elig, comp in zip(eligibility, completion_matrix): # exclude the last one
      candidate_mask = np.logical_and(elig, ~comp) # not done and eligible
      #candidate_mask = ~comp # not done and eligible
      if candidate_mask.sum() == 0:
        score = (~comp).astype(float)
      else:
        score = np.zeros((num_subtask), dtype=float)
        score[candidate_mask] = eligibility_trace[candidate_mask]
        #score[candidate_mask] = eligibility_trace[candidate_mask] + elig[candidate_mask].astype(float)
      if score.sum() == 0:
        import ipdb; ipdb.set_trace()
      prob = score / score.sum()
      probs.append(prob)

      ever_elig = np.logical_or(ever_elig, elig)
      eligibility_trace[ever_elig] = eligibility_trace[ever_elig] * _lambda
    return probs

  def compute_eligibility(self, completion):
    assert completion.ndim == 2
    assert completion.dtype == bool
    # self.ORmat # [num_subtask, num_and]
    num_data, num_subtask = completion.shape
    #import ipdb; ipdb.set_trace()

    completion_pm = completion * 2 - 1
    b_OR = np.expand_dims(self.ORmat.sum(axis=1) == 0, axis=0) # [1, num_subtask]
    #ANDout = (np.matmul(self.ANDmat, completion_pm)-b_AND+1).sign().ne(-1).int() #sign(A x indic + b) (+1 or 0)
    #eligibility = (np.matmul(self.ORmat, ANDout)-b_OR+0.5).sign().ne(-1).squeeze(-1)

    and_num_agreement = np.matmul(completion_pm, self.ANDmat.transpose()) # [num_data, num_subtask] x [num_and, num_subtask]^T = [num_data, num_and]
    ANDmax = abs(self.ANDmat).sum(axis=1, keepdims=True) # [num_and, 1]
    is_and_satisfied = and_num_agreement == ANDmax.transpose() # [num_data, num_and]
    eligibility = np.matmul(is_and_satisfied, self.ORmat.transpose()) + b_OR > 0 # [num_data, num_subtask]
    return eligibility

def get_gt_graph(task_name, path):
  gt_data = np.load(path, allow_pickle=True).item()
  graph_dict = gt_data[task_name]
  graph = SubtaskGraph(ANDmat=graph_dict['ANDmat'], ORmat=graph_dict['ORmat'], tind_by_layer=graph_dict['tind_by_layer'])
  return graph


def make_so_graph_dict(subtasks, options, so_sop, combined_effect_mat):
  N = len(subtasks)
  M = len(options)
  stacked_and = [sop for sop in so_sop if isinstance(sop, np.ndarray)]
  onehot = np.eye(M)
  stacked_or = [onehot[i] for i in range(M) if isinstance(so_sop[i], np.ndarray) for j in range(len(so_sop[i]))]
  if len(stacked_and) == 0:
    and_mat = None
  else:
    and_mat = np.concatenate(stacked_and).astype(np.float_)
  if len(stacked_or) == 0:
    or_mat = None
  else:
    or_mat = np.stack(stacked_or).T
  if and_mat is not None and or_mat is not None:
    assert and_mat.shape[0] == or_mat.shape[1]
  hard_effect_mat = np.zeros((M, N))
  soft_effect_mat = np.zeros((M, N))
  eps = 1e-4
  hard_effect_mat[combined_effect_mat[..., 1] > 1-eps] = 1
  hard_effect_mat[combined_effect_mat[..., 0] > 1-eps] = -1
  soft_effect_mat[(eps < combined_effect_mat[..., 1]) & (combined_effect_mat[..., 1] < 1-eps)] = 1
  soft_effect_mat[(eps < combined_effect_mat[..., 0]) & (combined_effect_mat[..., 0] < 1-eps)] = -1

  hard_should = np.zeros((M, N))
  soft_should = np.zeros((M, N))

  return {
      'Subtask_list': subtasks,
      'Option_list': options,
      'CAN': {
          'AndMat': and_mat,
          'OrMat': or_mat,
      },
      'SHOULD': {
          'soft_should': soft_should,
          'hard_should': hard_should,
      },
      'Effect': {
          'soft_effect': soft_effect_mat.T,
          'hard_effect': hard_effect_mat.T,
          'combined_effect': combined_effect_mat
      },
      'SOP': so_sop
  }

def get_graph_sop(so_graph, subtask_list=None, option_list=None, empty_value=True):
  and_mat = so_graph['CAN']['AndMat']
  or_mat = so_graph['CAN']['OrMat']
  so_graph['Subtask_list'] = [s.replace('system','SYSTEM').replace('user','USER') for s in so_graph['Subtask_list']]
  so_graph['Option_list'] = [s.replace('system','SYSTEM').replace('user','USER') for s in so_graph['Option_list']]
  #subtask_list = [s.replace('SYSTEM','system').replace('USER','user') for s in subtask_list]
  #option_list = [s.replace('SYSTEM','system').replace('USER','user') for s in option_list]
  if subtask_list is not None:
    if len(set(subtask_list) - set(so_graph['Subtask_list'])) > 0:
      print('Error: set of subtasks in "so_graph" and "subtask_list" are different.')
      print_list_1 = list(set(subtask_list))
      print_list_1.sort()
      print("subtask_list=", print_list_1)
      print_list_2 = list(set(so_graph['Subtask_list']))
      print_list_2.sort()
      print("subtask list in graph=", print_list_2)
      print("diff=", set(subtask_list) - set(so_graph['Subtask_list']))
      assert False
    inv_index = [so_graph['Subtask_list'].index(s) for s in subtask_list]
    if and_mat is not None:
      and_mat = and_mat[:, inv_index]
  else:
    subtask_list = so_graph['Subtask_list']

  if option_list is not None:
    assert len(set(option_list) - set(so_graph['Option_list'])) == 0
    inv_index = [so_graph['Option_list'].index(s) for s in option_list]
    if or_mat is not None:
      or_mat = or_mat[inv_index]
  else:
    option_list = so_graph['Option_list']

  N, M = len(subtask_list), len(option_list)
  if and_mat is None and or_mat is None:
    sop = [empty_value for _ in range(M)]
  else:
    assert and_mat.shape == (or_mat.shape[1], N)
    assert or_mat.shape == (M, and_mat.shape[0])
    sop = []
    eps = 1e-4
    for op in range(M):
      if or_mat[op].sum() > eps:
        sop.append(and_mat[or_mat[op] > eps].astype(np.int_))
      else: # if there is no AND node at all
        #sop.append(np.zeros((0, N)))
        sop.append(empty_value)
  return sop

def validate_subtask_graph(graph):
  keys = ['ANDmat', 'ORmat', 'tind_by_layer', 'W_a', 'W_o']
  assert all([key in graph for key in keys]), f"missing key! expected: {keys} and input: {graph.keys()}"

  ANDmat = graph['ANDmat']
  ORmat = graph['ORmat']
  W_a = graph['W_a']
  W_o = graph['W_o']
  tind_by_layer = graph['tind_by_layer']

  # tind_by_layer
  tind_list = []
  for curr_tind_list in tind_by_layer:
    tind_list = tind_list + curr_tind_list
  assert len(set(tind_list)) == max(tind_list) + 1, "error in tind_by_layer"

  # ANDmat <-> W_a
  num_layers = len(W_a)
  if 'and_layer_idx' in graph.keys():
    and_layer_idx = graph['and_layer_idx']
  else:
    num_and_by_layer = [wa.shape[0] for wa in W_a]
    bias=0
    and_layer_idx = []
    for num_and in num_and_by_layer:
      and_layer_idx.append([*range(bias, bias+num_and)])
      bias = bias + num_and
  #
  or_index_bias = 0
  and_index_bias = 0
  recon_andmat = np.zeros_like(ANDmat)
  recon_ormat = np.zeros_like(ORmat)
  for i in range(num_layers):
    num_and_nodes_in_current_layer = len(and_layer_idx[i])
    # OR -> AND (ANDmat & W_a)
    wa = W_a[i]
    current_or_indices = tind_by_layer[i]
    next_and_index_bias = and_index_bias + num_and_nodes_in_current_layer
    #
    recon_andmat[and_index_bias:next_and_index_bias, current_or_indices] = wa

    # AND -> OR (ORmat & W_o)
    wo = W_o[i]
    current_or_indices = tind_by_layer[i+1]
    recon_ormat[current_or_indices, and_index_bias:next_and_index_bias] = wo
    and_index_bias = next_and_index_bias

  assert np.all(recon_andmat==ANDmat), "mismatch between ANDmat and W_a"
  assert np.all(recon_ormat==ORmat), "mismatch between ORmat and W_o"

class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

class OptionSubtaskGraphVisualizer:
  def __init__(self):
    pass

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved graph @', path)
    return self

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'BT'
    dot.attr(nodesep="0.2", ranksep="0.3")
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  SUBTASK_NODE_STYLE = dict(shape='rect', height="0.2", width="0.2", margin="0.03")
  FEATURE_NODE_STYLE = dict(shape='rect', height="0.2", width="0.2", margin="0.03", rank="min")
  OPTION_NODE_STYLE = dict(shape='rect', height="0.2", width="0.2", margin="0.03")
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize_adjacency(self, adjacency: np.ndarray, subtask_label: List[str]):
    num_subtask = len(subtask_label)
    dot = self.make_digraph()
    for i1 in range(num_subtask):
      dot.node('SUBTASK'+str(i1), subtask_label[i1], **self.SUBTASK_NODE_STYLE)
    for i1 in range(num_subtask):
      for i2 in range(num_subtask):
        if adjacency[i1, i2]:
          dot.edge('SUBTASK'+str(i1), 'SUBTASK'+str(i2))
    return dot

  def visualize(self, cond_sop_by_subtask, effect_mat,
      subtask_label: List[str], option_label: List[str],
      hide_user=False, hide_system=False) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_sop_by_subtask: A sequence of eligibility CNF notations.
        cond_sop_by_subtask[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    dot = self.make_digraph()
    omitted_option_count = 0
    for ind, label in enumerate(option_label):
      subtask_indices = effect_mat[ind].nonzero()[0]
      sop_tensor = cond_sop_by_subtask[ind]
      if isinstance(sop_tensor, np.ndarray) or len(subtask_indices)> 0:
        if label.startswith('USER'):
          fill = '#a4c2f4ff'# light blue
        elif label.startswith('SYSTEM'):
          fill = '#f9cb9cff'# light orange
        else:
          fill = None
        dot.node('OPTION'+str(ind), label, fillcolor=fill, style='filled', **self.OPTION_NODE_STYLE) # visualize option node that has either effect or precondition
      else:
        omitted_option_count += 1
    print(f'#omitted options in the visualization={omitted_option_count}')

    for ind, label in enumerate(option_label):
      sop_tensor = cond_sop_by_subtask[ind]
      if hide_user and 'USER' in label:
        continue
      if hide_system and 'SYSTEM' in label:
        continue
      if isinstance(sop_tensor, np.ndarray):
        numA, feat_dim = sop_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", **self.OPERATOR_NODE_STYLE)
          # OR-AND
          dot.edge(anode_name, 'OPTION'+str(ind))
          sub_indices = sop_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            if subtask_label[sub_ind].startswith('f'):
              style = self.FEATURE_NODE_STYLE
            else:
              style = self.SUBTASK_NODE_STYLE
            dot.node('SUBTASK'+str(sub_ind), subtask_label[sub_ind], **style)
            sub_ind = sub_ind.item()
            target = 'SUBTASK'+str(sub_ind)

            if sop_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")

    for option_ind, option_label in enumerate(option_label):
      subtask_indices = effect_mat[option_ind].nonzero()[0]
      for subtask_ind in subtask_indices:
        from_node = 'OPTION'+str(option_ind)
        to_node = 'SUBTASK'+str(subtask_ind)
        if subtask_label[subtask_ind].startswith('f'):
          style = self.FEATURE_NODE_STYLE
        else:
          style = self.SUBTASK_NODE_STYLE
        dot.node('SUBTASK'+str(subtask_ind), subtask_label[subtask_ind], **style)
        if effect_mat[option_ind, subtask_ind, 1] > 0.99:
          dot.edge(from_node, to_node)
        elif effect_mat[option_ind, subtask_ind, 1] > 0.01:
          dot.edge(from_node, to_node, color='gray', label=f'{effect_mat[option_ind, subtask_ind, 1]*100:.2f}%')

        if effect_mat[option_ind, subtask_ind, 0] > 0.99:
          dot.edge(from_node, to_node, style="dashed")
        elif effect_mat[option_ind, subtask_ind, 0] > 0.01:
          dot.edge(from_node, to_node, style="dashed", color='gray', label=f'{effect_mat[option_ind, subtask_ind, 0]*100:.2f}%')

    return dot

  def visualize_andor(self, andmat, ormat, effect_mat,
      subtask_label: List[str], option_label: List[str],
      hide_user=False, hide_system=False) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_sop_by_subtask: A sequence of eligibility CNF notations.
        cond_sop_by_subtask[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    subtask_inds_used = []
    dot = self.make_digraph()
    omitted_option_count = 0
    # Add option nodes
    for ind, label in enumerate(option_label):
      subtask_indices = effect_mat[ind].nonzero()[0]
      orvec = ormat[ind]
      if orvec.sum() > 0:
        if label.startswith('USER'):
          fill = '#a4c2f4ff'# light blue
        elif label.startswith('SYSTEM'):
          fill = '#f9cb9cff'# light orange
        else:
          fill = None
        dot.node('OPTION'+str(ind), label, fillcolor=fill, style='filled', **self.OPTION_NODE_STYLE) # visualize option node that has either effect or precondition
      else:
        omitted_option_count += 1
    print(f'#omitted options in the visualization={omitted_option_count}')
    
    # Add and nodes
    num_a = andmat.shape[0]
    for ind in range(num_a):
      dot.node('AND'+str(ind), "&", **self.OPERATOR_NODE_STYLE)
      
    # Add option -> AND (precondition)
    for ind, label in enumerate(option_label):
      and_list = ormat[ind].nonzero()[0].tolist()
      for aind in and_list:
        dot.edge('AND'+str(aind), 'OPTION'+str(ind))
    
    # Add AND -> subtask (precondition)
    for aind in range(num_a):
      sub_list = andmat[aind].nonzero()[0].tolist()
      for sub_ind in sub_list:
        # Add subtask if non-exist
        if sub_ind not in subtask_inds_used:
          style = self.FEATURE_NODE_STYLE if subtask_label[sub_ind].startswith('f') else self.SUBTASK_NODE_STYLE
          dot.node('SUBTASK'+str(sub_ind), subtask_label[sub_ind], **style)
          subtask_inds_used.append(sub_ind)
        # Add edge
        style="dashed" if andmat[aind, sub_ind] > 0 else "solid"
        dot.edge("SUBTASK"+str(sub_ind), "AND"+str(aind), style=style)
    
    # subtask -> option (effect)
    for option_ind, label in enumerate(option_label):
      subtask_indices = effect_mat[option_ind].nonzero()[0]
      for subtask_ind in subtask_indices:
        from_node = 'OPTION'+str(option_ind)
        to_node = 'SUBTASK'+str(subtask_ind)
        if subtask_label[subtask_ind].startswith('f'):
          style = self.FEATURE_NODE_STYLE
        else:
          style = self.SUBTASK_NODE_STYLE
        dot.node('SUBTASK'+str(subtask_ind), subtask_label[subtask_ind], **style)
        if effect_mat[option_ind, subtask_ind, 1] > 0.99:
          dot.edge(from_node, to_node)
        elif effect_mat[option_ind, subtask_ind, 1] > 0.01:
          dot.edge(from_node, to_node, color='gray', label=f'{effect_mat[option_ind, subtask_ind, 1]*100:.2f}%')

        if effect_mat[option_ind, subtask_ind, 0] > 0.99:
          dot.edge(from_node, to_node, style="dashed")
        elif effect_mat[option_ind, subtask_ind, 0] > 0.01:
          dot.edge(from_node, to_node, style="dashed", color='gray', label=f'{effect_mat[option_ind, subtask_ind, 0]*100:.2f}%')

    return dot

  def _count_children(self, sub_indices, target_indices):
    count = 0
    for sub_ind in sub_indices:
      sub_ind = sub_ind.item()
      if sub_ind < 39 and sub_ind in target_indices:
        count += 1
    return count

class EffectGraphVisualizer:
  def __init__(self):
    pass

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(nodesep="0.2", ranksep="0.3")
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved effect graph @', path)
    return self

  SUBTASK_NODE_STYLE = dict(shape='oval', height="0.2", width="0.2", margin="0")
  FEATURE_NODE_STYLE = dict(shape='oval', height="0.2", width="0.2", margin="0", rank="min")
  OPTION_NODE_STYLE = dict(shape='rect', height="0.2", width="0.2", margin="0.03")
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize_soft(self, effect_mat,
      subtask_label: List[str], option_label: List[str]) -> 'graphviz.Digraph':
    assert effect_mat.shape == (len(option_label), len(subtask_label), 2)
    dot = self.make_digraph()
    for op_id, option in enumerate(option_label):
      for st_id, subtask in enumerate(subtask_label):
        if effect_mat[op_id, st_id].sum() > 0:
          dot.node('OP'+option, option, **self.OPTION_NODE_STYLE)
          dot.node('ST'+subtask, subtask, **self.SUBTASK_NODE_STYLE)
        if effect_mat[op_id, st_id, 0] > 1 - 1e-2:
          dot.edge('OP'+option, 'ST'+subtask, color="red")
        elif effect_mat[op_id, st_id, 0] > 1e-2:
          dot.edge('OP'+option, 'ST'+subtask, style='dashed', label=f'{effect_mat[op_id, st_id, 0]*100:.2f}%', color="red")

        if effect_mat[op_id, st_id, 1] > 1 - 1e-2:
          dot.edge('OP'+option, 'ST'+subtask)
        elif effect_mat[op_id, st_id, 1] > 1e-2:
          dot.edge('OP'+option, 'ST'+subtask, style='dashed', label=f'{effect_mat[op_id, st_id, 1]*100:.2f}%')
    return dot

def expand_sop(compact_sop_set, tind_by_layer, subtask_layer):
  num_subtask = len(subtask_layer)
  sop_set = [None] * num_subtask
  subtask_candidates = tind_by_layer[0]
  for layer_ind, subtasks_in_current_layer in enumerate(tind_by_layer):
    if layer_ind == 0:
      continue
    for subtask_index in subtasks_in_current_layer:
      compact_sop = compact_sop_set[subtask_index]
      num_and = compact_sop.shape[0]
      sop = np.zeros((num_and, num_subtask), dtype=int)
      np.put_along_axis(arr=sop, indices=np.array([subtask_candidates]), values=compact_sop, axis=1)
      sop_set[subtask_index] = sop
    subtask_candidates = subtask_candidates + subtasks_in_current_layer
  return sop_set

class GraphEvaluator:
  def __init__(self):
    pass

  def _validate(self, sop_infer_set, sop_gt_set):
    num_subtask = len(sop_gt_set)
    assert len(sop_infer_set) == num_subtask, f"length does not match. len(sop_infer_set)={len(sop_infer_set)}, len(sop_gt_set)={len(sop_gt_set)}"

    for sop in sop_gt_set:
      if isinstance(sop, np.ndarray):
        assert num_subtask == sop.shape[-1], f"GT sop shape does not match: num_subtask={num_subtask} != sop.shape[-1]={sop.shape[-1]}"
      else:
        assert sop is None or isinstance(sop, bool), f"expected None or bool but got {sop} as GT sop"

    for sop in sop_infer_set:
      if isinstance(sop, np.ndarray):
        assert num_subtask == sop.shape[-1], f"Inferred sop shape does not match: num_subtask={num_subtask} != sop.shape[-1]={sop.shape[-1]}"
      else:
        assert sop is None or isinstance(sop, bool), f"expected None or bool but got {sop} as inferred sop"

  def eval_graph(self, sop_infer_set, sop_gt_set, dump_path):
    #self._validate(sop_infer_set, sop_gt_set)
    # TP =  gt ^ infer
    # FP = ~gt ^ infer
    # FN =  gt ^ ~infer
    # TN = ~gt ^ ~infer
    num_subtask = len(sop_infer_set)
    precision, recall = np.zeros( (2,num_subtask) )
    for ind in range(num_subtask):
      sop_infer = sop_infer_set[ind] # each row is AND vector.
      sop_gt = sop_gt_set[ind]
      if sop_infer is None:
        precision[ind] = 1/pow(2, self._num_nonzero(sop_gt) )
        recall[ind] = 1.
        continue
      if isinstance(sop_gt, bool):
        #assert sop_infer.shape[1] < 30, "Too many subtasks! Remove DC columns before computing recall"
        if sop_gt:
          precision[ind] = 1.
          recall[ind] = self._count_sop_assign(sop_infer) / pow(2,  sop_infer.shape[1])
          continue
        else:
          precision[ind] = 0.
          recall[ind] = 0.5
          continue

      sop_infer_, sop_gt_ = self._compact(sop_infer, sop_gt)
      if sop_infer_ is None or sop_gt_ is None: # exactly same
        precision[ind] = 1.
        recall[ind] = 1.
      elif sop_gt_.ndim==0:
        if sop_gt_.item()==0:
          precision[ind] = 1.
          recall[ind] = 0.5
        else:
          precision[ind] = 0.5
          recall[ind] = 1.
      else:
        """ Testing
        sop_test = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 1]])
        assert self._count_sop_assign(sop_test) == 10
        """
        num_gt      = self._count_sop_assign(sop_gt_)
        num_infer   = self._count_sop_assign(sop_infer_) # checked!

        # 1. TP
        TP = self._get_A_and_B( sop_infer_, sop_gt_ )
        FN = num_gt - TP
        FP = num_infer - TP
        precision[ind]  = TP / (TP+FP)
        recall[ind]     = TP / (TP+FN)

    filename = os.path.join(dump_path, 'graph_PR.txt')
    # with open(filename, 'a') as f:
      # string = '{}\t{:.02f}\t{:.02f}\t'.format(0, precision.mean(), recall.mean())
      # f.writelines( string )
    return precision, recall

  def _compact(self, sop1, sop2):
      if isinstance(sop1, bool):
          if sop1==sop2:
            return None, None
          else:
            return sop1, sop2
      if sop1.shape == sop2.shape:
        diff_count = (sop1!=sop2).sum(0) # num diff elems in each dimension
        if diff_count.sum()==0:
            return None, None
        else:
          indices = diff_count.nonzero()[0].squeeze()
          return sop1[:,indices], sop2[:, indices]
        #data = np.concatenate( (sop1, sop2.expand_dim), dim= )
      else:
        return sop1, sop2

  def _get_A_and_B(self, sop1, sop2):
    if sop1.ndim == 1:
      sop1 = np.expand_dims(sop1, 0)
    if sop2.ndim == 1:
      sop2 = np.expand_dims(sop2, 0)

    sop_list = []
    for andvec1 in sop1:
      for andvec2 in sop2:
        and_mul = andvec1*andvec2 # vec * vec (elem-wise-mul)
        if (and_mul==-1).sum() > 0: # if there exists T^F--> empty (None)
          continue
        and_sum = andvec1+andvec2
        #    |  1 |  0 | -1
        #----+----------------
        #  1 |  1    1   None
        #----|
        #  0 |  1    0   -1
        #----|
        # -1 | None -1   -1
        and_sum = and_sum.clip(min=-1, max=1)
        sop_list.append(and_sum.squeeze())
    if len(sop_list)>0:
      sop_mat = np.stack(sop_list, axis=0)
      return self._count_sop_assign( sop_mat )
    else:
      return
  """
  def _get_A_and_B(self, sop1, sop2):
    assert sop2.ndim == 1 or sop2.shape[0] == 1, f"sop2 is not a vector! sop2.shape={sop2.shape}"
    numA = sop1.shape[0]
    sop_list = []
    #1. merge AND
    for aind in range(numA):
        and1 = sop1[aind]
        and_mul = and1*sop2 # vec * vec (elem-wise-mul)
        if (and_mul==-1).sum() > 0: # if there exists T^F--> empty (None)
            continue
        and_sum = and1+sop2
        #    |  1 |  0 | -1
        #----+----------------
        #  1 |  1    1   None
        #----|
        #  0 |  1    0   -1
        #----|
        # -1 | None -1   -1
        and_sum = and_sum.clip(min=-1, max=1)
        sop_list.append(and_sum.squeeze())
    if len(sop_list)>0:
        sop_mat = np.stack(sop_list, axis=0)
        return self._count_sop_assign( sop_mat )
    else:
        return"""

  def _count_sop_assign(self, sop_mat):
      # count the number of binary combinations that satisfy input 'sop_mat'
      # where each row sop_mat[ind, :] is a condition, and we take "or" of them.
      if sop_mat.ndim==0:
          return 1
      elif sop_mat.ndim==1 or sop_mat.shape[0]==1: # simply count number of 0's
          return pow(2, self._num_zero(sop_mat) )
      NA, dim = sop_mat.shape

      # 0. prune out all-DC bits
      target_indices = []
      common_indices = []
      for ind in range(dim):
          if not np.all(sop_mat[:, ind]==sop_mat[0, ind]):
              target_indices.append(ind)
          else: # if all the same, prune out.
              common_indices.append(ind)
      common_sop = sop_mat[0,common_indices] # (dim)
      num_common = pow(2, self._num_zero(common_sop) )

      compact_sop = sop_mat[:, target_indices].astype(np.int8)
      numO, feat_dim = compact_sop.shape # NO x d

      if feat_dim > 25:
          print('[Warning] there are too many non-DC features!! It will take long time')
          import ipdb; ipdb.set_trace()

      # 1. gen samples
      nelem = pow(2, feat_dim)
      bin_mat = np.array(list(map(list, itertools.product([0, 1], repeat=feat_dim))), dtype=bool) # +1 / -1
      # N x d
      false_mat = ~bin_mat
      false_map = (np.expand_dims(compact_sop, axis=0) * np.expand_dims(false_mat, axis=1)).sum(2) # NxNOxd --> NxNO
      true_map = (false_map==0) # NxNO
      truth_val_vec = (true_map.sum(1)>0) # if any one of NO is True, then count as True (i.e, OR operation)
      return truth_val_vec.sum().item() * num_common

  def _num_nonzero(self, sop):
    if isinstance(sop, bool):
      return 1
    elif sop is None:
      return 0
    else:
      return (sop!=0).sum().item()

  def _num_zero(self, sop):
      return (sop==0).sum().item()

class GraphVisualizer:
  def __init__(self):
    pass

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved graph @', path)
    return self

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    #dot.graph_attr['rank'] = 'min'
    dot.attr(nodesep="0.2", ranksep="0.3", rank='same')
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  SUBTASK_NODE_STYLE = dict()
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize_adjacency(self, adjacency: np.ndarray, subtask_label: List[str]):
    num_subtask = len(subtask_label)
    dot = self.make_digraph()
    for i1 in range(num_subtask):
      dot.node('SUBTASK'+str(i1), subtask_label[i1], **self.SUBTASK_NODE_STYLE)
    for i1 in range(num_subtask):
      for i2 in range(num_subtask):
        if adjacency[i1, i2]:
          dot.edge('SUBTASK'+str(i1), 'SUBTASK'+str(i2))
    return dot

  def visualize_logicgraph(self, g: 'mtsgi.envs.logic_graph.SubtaskLogicGraph'
                           ) -> 'graphviz.Digraph':
    import mtsgi.envs.logic_graph
    LogicOp = mtsgi.envs.logic_graph.LogicOp

    dot = self.make_digraph()
    def _visit_node(node: LogicOp, to: str, has_negation=False):
      # TODO: This access private properties of LogicOp too much.
      # definitely we should move this to logic_graph?
      if node._op_type == LogicOp.TRUE:
        #v_true = f'true_{id(node)}'
        #dot.edge(v_true, to, style='filled')
        pass
      elif node._op_type == LogicOp.FALSE:
        v_false = f'_false_'
        dot.edge(v_false, to, style='filled', shape='rect')
      elif node._op_type == LogicOp.LEAF:
        leaf = node._children[0]
        dot.edge(leaf.name, to, style=has_negation and 'dashed' or '')
      elif node._op_type == LogicOp.NOT:
        op: LogicOp = node._children[0]
        _visit_node(op, to=to, has_negation=not has_negation)
      elif node._op_type == LogicOp.AND:
        v_and = f'and_{to}_{id(node)}'
        dot.node(v_and, "&", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_and, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_and)
        pass
      elif node._op_type == LogicOp.OR:
        v_or = f'or_{to}_{id(node)}'
        dot.node(v_or, "|", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_or, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_or)
      else:
        assert False, str(node._op_type)

    for name, node in g._nodes.items():
      assert name == node.name
      dot.node(node.name)
      _visit_node(node.precondition, to=name)
    return dot

  def visualize(self, cond_sop_by_subtask, subtask_layer,
              subtask_label: List[str], show_index=False) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_sop_by_subtask: A sequence of eligibility CNF notations.
        cond_sop_by_subtask[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    for sop in cond_sop_by_subtask:
      if sop is not None:
        assert sop.shape[1] == len(subtask_label), "input cond_sop_by_subtask has different shape per subtask"
    dot = self.make_digraph()
    num_subtasks = len(subtask_label)

    #cond_sop_by_subtask
    # Add ORnodes by layer
    for layer_ind in range(subtask_layer.max()+1):
      candidates = [i for i, layer in enumerate(subtask_layer) if layer == layer_ind]
      if len(candidates) > 0:
        with dot.subgraph() as s:
          s.attr(rank='same')
          for subtask_index in candidates:
            label = f"{subtask_index+1}.{subtask_label[subtask_index]}"
            s.node('OR'+str(subtask_index), label, shape='rect', height="0.2", width="0.2", margin="0")

    for ind in range(num_subtasks):
      if subtask_layer[ind] > -2:
        #label = subtask_label[ind]
        #dot.node('OR'+str(ind), label, shape='rect', height="0.2", width="0.2", margin="0")
        sop_tensor = cond_sop_by_subtask[ind]
        if sop_tensor is None or (sop_tensor.shape[0] == 1 and sop_tensor.sum()==0): # no precondition
          # None: always elig or never elig
          # shape == (1, ): only happens when data is noisy.
          continue
        numA, feat_dim = sop_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", shape='rect', style='filled',
                   height="0.15", width="0.15", margin="0.03")
          # OR-AND
          dot.edge(anode_name, 'OR'+str(ind))
          sub_indices = sop_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            target = 'OR'+str(sub_ind)

            if sop_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")
    return dot

  def visualize_gt(self, cond_sop_by_subtask, subtask_layer,
              subtask_label: List[str], show_index=False) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_sop_by_subtask: A sequence of eligibility CNF notations.
        cond_sop_by_subtask[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    for i in range(len(cond_sop_by_subtask)):
      if type(cond_sop_by_subtask[i]) == bool:
        cond_sop_by_subtask[i] = None
    for sop in cond_sop_by_subtask:
      if sop is not None:
        assert sop.shape[1] == len(subtask_label), "input cond_sop_by_subtask has different shape per subtask"
    dot = self.make_digraph()
    num_subtasks = len(subtask_label)

    #cond_sop_by_subtask
    # Add ORnodes by layer
    for layer_ind in range(subtask_layer.max()+1):
      candidates = [i for i, layer in enumerate(subtask_layer) if layer == layer_ind]
      if len(candidates) > 0:
        with dot.subgraph() as s:
          s.attr(rank='same')
          for subtask_index in candidates:
            label = f"{subtask_index+1}.{subtask_label[subtask_index]}"
            s.node('OR'+str(subtask_index), label, shape='rect', height="0.2", width="0.2", margin="0")

    for ind in range(num_subtasks):
      if subtask_layer[ind] > -2:
        #label = subtask_label[ind]
        #dot.node('OR'+str(ind), label, shape='rect', height="0.2", width="0.2", margin="0")
        sop_tensor = cond_sop_by_subtask[ind]
        if sop_tensor is None or (sop_tensor.shape[0] == 1 and sop_tensor.sum()==0): # no precondition
          # None: always elig or never elig
          # shape == (1, ): only happens when data is noisy.
          continue
        numA, feat_dim = sop_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", shape='rect', style='filled',
                   height="0.15", width="0.15", margin="0.03")
          # OR-AND
          dot.edge(anode_name, 'OR'+str(ind))
          sub_indices = sop_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            target = 'OR'+str(sub_ind)

            if sop_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")
    return dot

  def save_and_or(self, cond_sop_by_subtask, subtask_layer,
                subtask_label: List[str], show_index=False, task_name = ' ') -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_sop_by_subtask: A sequence of eligibility CNF notations.
        cond_sop_by_subtask[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    import pickle
    for i in range(len(cond_sop_by_subtask)):
      if type(cond_sop_by_subtask[i]) == bool:
        cond_sop_by_subtask[i] = None

    for sop in cond_sop_by_subtask:
      if sop is not None:
        assert sop.shape[1] == len(subtask_label), "input cond_sop_by_subtask has different shape per subtask"
    dot = self.make_digraph()
    num_subtasks = len(subtask_label)

    or_nodes = []
    and_nodes = []
    edges = []
    #cond_sop_by_subtask
    # Add ORnodes by layer
    for layer_ind in range(subtask_layer.max()+1):
      candidates = [i for i, layer in enumerate(subtask_layer) if layer == layer_ind]
      if len(candidates) > 0:
        with dot.subgraph() as s:
          s.attr(rank='same')
          for subtask_index in candidates:
            label = f"{subtask_index+1}.{subtask_label[subtask_index]}"
            s.node('OR'+str(subtask_index), label, shape='rect', height="0.2", width="0.2", margin="0")

            or_nodes.append('OR'+str(subtask_index)+label)

    for ind in range(num_subtasks):
      if subtask_layer[ind] > -2:
        #label = subtask_label[ind]
        #dot.node('OR'+str(ind), label, shape='rect', height="0.2", width="0.2", margin="0")
        sop_tensor = cond_sop_by_subtask[ind]
        if sop_tensor is None or (sop_tensor.shape[0] == 1 and sop_tensor.sum()==0): # no precondition
          # None: always elig or never elig
          # shape == (1, ): only happens when data is noisy.
          continue
        numA, feat_dim = sop_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", shape='rect', style='filled',
                   height="0.15", width="0.15", margin="0.03")
          and_nodes.append(anode_name)
          # OR-AND
          dot.edge(anode_name, 'OR'+str(ind))
          sub_indices = sop_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            target = 'OR'+str(sub_ind)

            edges.append((target,anode_name))
            if sop_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")

    node_num = len(or_nodes)
    edge_num = len(and_nodes)
    and_graph = np.zeros([edge_num, node_num], dtype=bool)
    or_graph = np.zeros([node_num, edge_num], dtype=bool)
    and_node_ind = {}
    for i in range(len(and_nodes)):
      and_node_ind[int(and_nodes[i][3:][:-2])] = i
    for i in range(len(edges)):
      left = int(edges[i][0][2:])
      right = int(edges[i][1][3:][:-2])
      idx = and_node_ind[right]
      and_graph[idx, left] = True
      or_graph[right, idx] = True
    new_graph = {
      'and': and_graph,
      'or': or_graph,
    }
    andor_graphs = pickle.load(open('./procel_andor_matrix_v2.pkl','rb'))
    andor_graphs[task_name]=new_graph
    pickle.dump(andor_graphs, open('./procel_andor_matrix_v2.pkl','wb'))
    embed()
    exit()
    # andor_graphs = pickle.load(open('./our_procel_andor_matrix_v2.pkl','rb'))
    # andor_graphs[task_name]=new_graph
    # pickle.dump(andor_graphs, open('./our_procel_andor_matrix_v2.pkl','wb'))
    # andor_graphs = pickle.load(open('./visonly_procel_andor_matrix_v2.pkl','rb'))
    # andor_graphs[task_name]=new_graph
    # pickle.dump(andor_graphs, open('./visonly_procel_andor_matrix_v2.pkl','wb'))
    return dot

  def _count_children(self, sub_indices, target_indices):
    count = 0
    for sub_ind in sub_indices:
      sub_ind = sub_ind.item()
      if sub_ind < 39 and sub_ind in target_indices:
        count += 1
    return count

# Old utils

def sample_subtasks(
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int,
    maximum_size: Optional[int] = None,
    replace: bool = False
) -> List[str]:
  if maximum_size is not None:
    assert maximum_size <= len(pool), 'Invalid maximum_size.'
  maximum_size = maximum_size or len(pool)
  random_size = rng.randint(minimum_size, maximum_size + 1)
  sampled_subtasks = rng.choice(pool, size=random_size, replace=replace)
  return list(sampled_subtasks)

def add_sampled_nodes(
    graph: 'logic_graph.SubtaskLogicGraph',
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int = 1,
    maximum_size: Optional[int] = None
):
  valid_nodes = list(graph.nodes)

  # Sample distractors.
  distractors = sample_subtasks(
      rng=rng,
      pool=pool,
      minimum_size=minimum_size,
      maximum_size=maximum_size
  )

  distractors_added = []
  for distractor in distractors:
    if distractor not in graph:
      distractor_at = np.random.choice(valid_nodes)
      graph[distractor] = distractor_at
      distractors_added.append(distractor)
  return graph, distractors_added


def _sktree_to_sop(dt, num_feats):
  tree = dt.tree_
  root = 0
  if tree.children_left[root] == -1:
    # is leaf node
    return None
  all_sop = []
  empty_sop = np.zeros(num_feats, dtype=np.int_)
  nodes = [(empty_sop, root)]
  while len(nodes) > 0:
    sop, node = nodes.pop()
    if tree.children_left[node] == -1:
      if tree.value[node][0].argmax():
        all_sop.append(sop)
      continue
    left_sop = sop.copy()
    right_sop = sop.copy()

    feature = tree.feature[node]
    left = tree.children_left[node]
    right = tree.children_right[node]
    left_sop[feature] = -1
    right_sop[feature] = 1

    nodes.append((left_sop, left))
    nodes.append((right_sop, right))
  if len(all_sop) == 0:
    return None
  return np.stack(all_sop)