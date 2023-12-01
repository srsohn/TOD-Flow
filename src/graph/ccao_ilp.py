"""Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List
import numpy as np

from graph.ao_reduce import AndOrReduction
from util.graph_utils import SubtaskGraph, dotdict, GraphVisualizer

MAX_GINI=2.0
LEFT_BIAS=5.0
ALLOW_NOISE_IN_DATA=True
MAX_COMPLEXITY = 2

class MSG2(AndOrReduction):
  @property
  def algo_name(self):
    return f"MSG2"

  def add_data(self, ilp_data=None, order_data=None):
    assert ilp_data is not None and order_data is not None, "Error: 'ilp_data' and 'order_data' are required!"
    self._parse_ilp_data(ilp_data)
    self._parse_order_data(order_data)
  
  def infer_task(self, edge_purity_threshold, weighted=False) -> List[SubtaskGraph]:
    #1. Infer precondition
    if weighted:
      weight_by_subtask = self.compact_weight_by_subtask
    else:
      weight_by_subtask = None
    sop_set, tind_by_layer, subtask_layer = self._infer_layerwise_precondition(self.compact_comp_by_subtask, edge_purity_threshold, weight_by_subtask)

    self._sop_set = sop_set
    return sop_set, tind_by_layer, subtask_layer

  def _infer_layerwise_precondition(self, completion_list, edge_purity_threshold, weight_by_subtask=None):
    # 0. assertions

    # Compute pseudo-layers from order data
    subtask_layer, precedents_by_subtask, cd_children_by_subtask, edge_purity = self.assign_pseudo_layer(self.pairwise_weighted_count_table, edge_purity_threshold=edge_purity_threshold)
    # cd_children_by_subtask: "critical-direct-children-by-subtask"
    first_layer_ind_list = np.argwhere(subtask_layer==0).squeeze(1).tolist() # Find subtasks in first layer

    # Initialize variables
    curr_layer_ind_list = first_layer_ind_list
    tind_by_layer = [curr_layer_ind_list]
    cand_ind_list = curr_layer_ind_list.copy()

    # 2. layer-wise AO-ILP
    sop_set = [None] * self._num_subtask
    pc_matrix = np.zeros((self._num_subtask, self._num_subtask), bool)
    for layer_ind in range(1, self._num_subtask):
      if len(cand_ind_list) == self._num_subtask: # Inference done.
        break
      max_complexity = min(MAX_COMPLEXITY, len(cand_ind_list))
      curr_layer_ind_list = []
      for ind in range(self._num_subtask):
        soft_elig_by_traj = self.soft_elig_by_traj_by_subtask[ind]
        soft_elig_by_traj = soft_elig_by_traj[soft_elig_by_traj.sum(1)>0, :]
        if subtask_layer[ind] != layer_ind:
          continue
        assert ind not in cand_ind_list, "error. already assigned"
        completion = completion_list[ind]
        assert completion.ndim == 2, f"error: completion.ndim == {completion.ndim}, {completion}"
        comp = completion[:, cand_ind_list]
        #
        cd_children = cd_children_by_subtask[ind]
        is_cd_children = np.array([True if i in cd_children else False for i in range(self._num_subtask)])
        full_sop_org = self.cc_cart_train(is_cd_children, cand_ind_list, pc_matrix, soft_elig_by_traj) # 88%
        compact_sop = self.compactify_sop(full_sop_org, cand_ind_list)
        #compact_sop_org = self.cc_cart_train_exhaustive(is_cd_children, pc_matrix, inputs=comp, cand_ind_list=cand_ind_list, max_complexity=max_complexity) # 88%

        assert full_sop_org.ndim == 2
        if full_sop_org.shape[0] == 1: # have no OR relation.
          subtask_idxs = full_sop_org.squeeze().nonzero()[0]
          pc_matrix[ind, subtask_idxs] = True
          for idx in subtask_idxs: # add children (=pc_matrix[idx, :]) of each child (=idx) to 'ind'
            # ind - idx - child
            pc_matrix[ind, :] = np.logical_or(pc_matrix[ind, :], pc_matrix[idx, :])
        
        assert compact_sop.ndim == 2, "sop_tensor should be 2 dimension"
        assert compact_sop.shape[1] == len(cand_ind_list), "sop_tensor's feature dimension is wrong"
        sop_set[ind] = compact_sop
        #
        curr_layer_ind_list.append(ind)

      assert len(curr_layer_ind_list) > 0, f"Error: layer-wise ILP failed to find precondition for subtasks {np.nonzero(subtask_layer==-1)}"
      cand_ind_list.extend(curr_layer_ind_list)
      tind_by_layer.append(curr_layer_ind_list)

    return sop_set, tind_by_layer, subtask_layer
  
  def compactify_sop(self, full_sop, cand_ind_list):
    return full_sop[:, cand_ind_list]

  def cc_cart_train(self, is_cd_children, cand_ind_list, pc_matrix, soft_elig_by_traj, complexity_penalty=0.2): # complexity constrained
    # inputs: list of completion vectors when target subtask was eligible
    # first-branching: among critical direct children (which is a subset of candidates), choose one with highest weighted accuracy (== \sum_i {e_i * I(data_i is predicted as positive)} / \sum_i {e_i})
    # later: among all candidates that are not the (grand-)child of the already chosen ones, choose one with highest weighted accuracy
    num_traj = soft_elig_by_traj.shape[0] # shape=[num_traj, num_subtask]
    cd_child_list = is_cd_children.nonzero()[0]

    """
    # Singleton (2/4 of entire set)
    soft_elig_singleton = soft_elig_by_traj.sum(0) / num_traj # shape=[num_subtask, ]. range(0, 1)
    cd_child_list = is_cd_children.nonzero()[0]
    ind = np.argmax(soft_elig_singleton[cd_child_list])
    best_subtask_idx1 = cd_child_list[ind]
    score1 = soft_elig_singleton[best_subtask_idx1]
    best_sop = np.zeros((1, self._num_subtask), dtype=int)
    best_sop[0, best_subtask_idx1] = 1"""

    ### Prep
    is_pc_matrix = np.logical_or(pc_matrix, pc_matrix.transpose())

    best_score = 0.
    for idx1 in cd_child_list:
      # singleton
      soft_elig = soft_elig_by_traj[:, idx1].sum() / num_traj
      best_single_score = soft_elig * 4/2
      if best_score < best_single_score:
        best_score = best_single_score
        best_sop = np.zeros((1, self._num_subtask), dtype=int)
        best_sop[0, idx1] = 1

      # pair
      for idx2 in cand_ind_list:
        if is_pc_matrix[idx1, idx2] == False and idx2 != idx1: # non-pc and diff
          #
          soft_elig_and = np.minimum(soft_elig_by_traj[:, idx1], soft_elig_by_traj[:, idx2]).sum() / num_traj
          best_and_score = soft_elig_and * 4 - complexity_penalty
          if best_score < best_and_score:
            best_score = best_and_score
            best_sop = np.zeros((1, self._num_subtask), dtype=int)
            best_sop[0, [idx1, idx2]] = 1
          #
          soft_elig_or = np.maximum(soft_elig_by_traj[:, idx1], soft_elig_by_traj[:, idx2]).sum() / num_traj
          best_or_score = soft_elig_or * 4/3 - complexity_penalty
          if best_score < best_or_score:
            best_score = best_or_score
            best_sop = np.zeros((2, self._num_subtask), dtype=int)
            best_sop[0, idx1] = 1
            best_sop[1, idx2] = 1

      if False:
        print(score1, score_and, score_or)
        print(best_sop)
    return best_sop

  def _cycle_exists(self, edge_matrix):
    mult_matrix = edge_matrix.copy()
    num_subtask = len(edge_matrix)
    for i in range(1, num_subtask):
      mult_matrix = np.matmul(mult_matrix, edge_matrix)
      if mult_matrix.trace() != 0:
        return True
    return False
  
  def assign_pseudo_layer(self, pairwise_count_table, edge_purity_threshold = 0.9):
    # assign layer based on relative order.
    # Rule: if #(A->B) / {#(A->B) + #(B->A)} > 0.9 -> then layer(A) < layer(B)
    assert edge_purity_threshold > 0.5, "edge_purity_threshold must be larger than 0.5"

    subtask_layer = np.zeros(self._num_subtask, dtype=int)
    precedents_by_subtask = [[] for _ in range(self._num_subtask) ]
    pairwise_count_ = pairwise_count_table + 1e-5
    pairwise_count_transpose = pairwise_count_.transpose()
    pairwise_occurence_count_ = pairwise_count_ + pairwise_count_transpose
    edge_purity = pairwise_count_ / pairwise_occurence_count_

    occur_count_thresh = 0.1
    #no_occurence_mask = pairwise_occurence_count_ < 2 # if it happened only once or never, don't add edge.
    no_occurence_mask = pairwise_occurence_count_ < occur_count_thresh # if it happened only once or never, don't add edge.
    edge_purity[no_occurence_mask] = 0.

    edge_matrix = edge_purity > edge_purity_threshold
    while self._cycle_exists(edge_matrix):
      print(f'Warning! cycle detected. increasing count threshold {occur_count_thresh+1}')
      occur_count_thresh = occur_count_thresh + 1
      no_occurence_mask = pairwise_occurence_count_ < occur_count_thresh # if it happened only once or never, don't add edge.
      edge_purity[no_occurence_mask] = 0.
      edge_matrix = edge_purity > edge_purity_threshold

    pair_idxs = np.argwhere(edge_matrix) # a -> b with 90% chance => layer(A) < layer(B)

    flag = True
    while flag:
      flag = False
      for idx1, idx2 in pair_idxs:
        if idx1 not in precedents_by_subtask[idx2]:
          precedents_by_subtask[idx2].append(idx1) # meaning idx1 --> idx2 is true but idx2 --> idx1 is (almost) never.
        if subtask_layer[idx1]+1 > subtask_layer[idx2]:
          subtask_layer[idx2] = subtask_layer[idx1]+1
          flag = True
    
    #import ipdb; ipdb.set_trace()
    # find critical-direct children
    cd_children_by_subtask = []
    for idx2 in range(self._num_subtask):
      cd_children = []
      for idx1 in precedents_by_subtask[idx2]: # critical
        if subtask_layer[idx1] == subtask_layer[idx2] - 1: # direct
          cd_children.append(idx1)
      cd_children_by_subtask.append(cd_children)
    if False: # debugging
      print('subtask-by-layer:')
      for layer_ind in range(subtask_layer.max()+1):
        print_str = ""
        for ind in range(self._num_subtask):
          if subtask_layer[ind] == layer_ind:
            print_str = print_str + f"[{ind}:{self._subtask_label[ind]}] / "
        print(print_str[:-3])
    return subtask_layer, precedents_by_subtask, cd_children_by_subtask, edge_purity

"""Unused
  def populate_candidates(self, cand_ind_list, complexity, is_cd_children, pc_matrix, force_direct_child=True, avoid_dr_children=True):
    from itertools import combinations
    subtask_idxs_candidates = combinations(cand_ind_list, complexity)
    subtask_idxs_candidates = np.array(list(subtask_idxs_candidates))

    idx_to_compact = {ind:i for i, ind in enumerate(cand_ind_list)}
    #mask = np.zeros((len(subtask_idxs_candidates)), dtype=bool)
    compact_idxs_candidates = []
    for i, subtask_idxs in enumerate(subtask_idxs_candidates):
      # if there is no direct child, skip it.
      if not np.any(is_cd_children[subtask_idxs]) and force_direct_child:
        continue
        
      if len(subtask_idxs) == 1:
        compact_idxs_candidates.append([idx_to_compact[subtask_idxs.item()]])
        continue

      # remove if (grand-)parent-(grand-)child relationship exists among children
      if avoid_dr_children:
        num_dc = pc_matrix[subtask_idxs, :][:, subtask_idxs].sum()
        if num_dc > 0:
          continue
      compact_idx_list = []
      for idx in subtask_idxs:
        compact_idx_list.append(idx_to_compact[idx])
      compact_idxs_candidates.append(compact_idx_list)
    
    return compact_idxs_candidates
  
  def cc_cart_train_exhaustive(self, is_cd_children, pc_matrix, inputs, cand_ind_list, max_complexity, complexity_penalty=0.8): # complexity constrained
    # inputs: list of completion vectors when target subtask was eligible
    # consider complexity 1 and 2.
    num_all, num_features = inputs.shape

    # Understanding scale
    best_score = - 100
    for complexity in range(1, max_complexity+1):
      # populate
      compact_idxs_candidates = self.populate_candidates(cand_ind_list, complexity, is_cd_children, pc_matrix, force_direct_child=True)
      for idxs in compact_idxs_candidates:
        if True:
          # AND
          num_pp = np.all(inputs[:, idxs], axis=1).sum() # sop (A&B) is true & elig is true
          scale = pow(2, complexity) # sop (A1|A2|..Ak) is 1/2^k * #total
          coverage = num_pp / num_all
          score_and = scale * coverage - complexity_penalty * complexity

          # OR
          num_pp = np.any(inputs[:, idxs], axis=1).sum() # sop (A|B) is true & elig is true
          scale = pow(2, complexity) / (pow(2, complexity) - 1) # sop (A1|A2|..Ak) is (2^k-1)/2^k * #total
          coverage = num_pp / num_all
          score_or = scale * coverage - complexity_penalty * complexity
          #print(f"idxs={idxs}, complexity={complexity}, score_and={score_and}, score_or={score_or}")
        else:
          # AND
          num_pp = np.all(inputs[:, idxs], axis=1).sum() # sop (A&B) is true & elig is true
          scale = pow(2, complexity) # sop (A1|A2|..Ak) is 1/2^k * #total
          miss_portion = 1 - num_pp / num_all + 1e-6
          neg_log_miss = - log2(miss_portion)
          score_and = scale * neg_log_miss - complexity_penalty * complexity

          # OR
          num_pp = np.any(inputs[:, idxs], axis=1).sum() # sop (A|B) is true & elig is true
          scale = pow(2, complexity) / (pow(2, complexity) - 1)# sop (A1|A2|..Ak) is (2^k-1)/2^k * #total
          miss_portion = 1 - num_pp / num_all + 1e-6
          neg_log_miss = - log2(miss_portion)
          score_or = scale * neg_log_miss - complexity_penalty * complexity

        
        if score_and > score_or:
          if best_score < score_and:
            best_idxs = idxs
            best_is_and = True
            best_score = score_and
        else:
          if best_score < score_or:
            best_idxs = idxs
            best_is_and = False
            best_score = score_or
    
    if best_is_and:
      # AND
      best_sop = np.zeros((1, num_features))
      best_sop[0, best_idxs] = 1
    else:
      # OR
      best_sop = np.zeros((len(best_idxs), num_features))
      for i, idx in enumerate(best_idxs):
        best_sop[i, idx] = 1
    #print('===================')
    #print('best=',best_score)
    return best_sop
    """
