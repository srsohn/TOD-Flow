"""Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List

import numpy as np
import queue

from util import graph_utils, logic_utils
from util.graph_utils import SubtaskGraph, dotdict

MAX_GINI=2.0
LEFT_BIAS=5.0
ALLOW_NOISE_IN_DATA=True

class AndOrReduction:

  def __init__(
      self,
      ilp_args: dict,
      # Dataset
      task_name: str = None,
      subtask_label: list = None,
      option_label: list = None,
      # 
      verbose: bool = False):
    # data parsing-related hparams
    
    # ilp algorithm hparams
    self._augment_negative = False
    
    # Dataset parameter
    self._task_name = task_name
    self._subtask_label = subtask_label
    self._option_label = option_label
    self._num_subtask = len(subtask_label) if subtask_label is not None else None
    self._num_option = len(option_label) if option_label is not None else None

    # Logging & debugging
    self._verbose = verbose

  @property
  def algo_name(self):
    return f"AOReduce"

  def add_data(self, ilp_data=None, order_data=None):
    assert ilp_data is not None, "Error: 'ilp_data' is required!"
    self._parse_ilp_data(ilp_data)

  def _parse_order_data(self, order_data):
    self.num_trajectories = order_data['num_trajectories']
    self.first_count = order_data['first_count']
    self.pairwise_count_table = order_data['precedent_count']
    self.pairwise_weighted_count_table = order_data['precedent_weighted_count']
    
  def _parse_ilp_data(self, ilp_data):
    self.comp_by_subtask = ilp_data['completion']
    self.elig_by_subtask = ilp_data['eligibility']
    self.weight_by_subtask = ilp_data['weight']
    self.precedent_set_list = ilp_data['precedent_set_list']
    self.soft_elig_by_traj_by_subtask = ilp_data['soft_elig_by_traj_by_subtask']

    if self._augment_negative:
      neg_comp = ilp_data['neg_completion']
      neg_elig = ilp_data['neg_eligibility']
      for subtask_idx in range(self._num_subtask):
        self.comp_by_subtask[subtask_idx].extend(neg_comp[subtask_idx])
        self.elig_by_subtask[subtask_idx].extend(neg_elig[subtask_idx])

    self.compact_comp_by_subtask, self.compact_weight_by_subtask = [], []
    for subtask_idx, completion in enumerate(self.comp_by_subtask):
      weight = np.array(self.weight_by_subtask[subtask_idx])
      if completion.ndim == 1: # no precondition
        self.compact_comp_by_subtask.append(completion)
        self.compact_weight_by_subtask.append(weight)
      else:
        compact_completion, mask = self.simplify_sop(completion)
        self.compact_comp_by_subtask.append(compact_completion)
        self.compact_weight_by_subtask.append(weight[mask])

  def infer_task(self) -> List[SubtaskGraph]:
    assert self._check_data(), "ilp data is invalid!"

    #1. Infer precondition
    sop_set, tind_by_layer, subtask_layer = self._infer_precondition_by_boolean_reduction(
        self.comp_by_subtask,
        self.elig_by_subtask
    )
    if tind_by_layer is None:
      print('Failed to run layer-wise ILP')

    self._sop_set = sop_set
    return sop_set, tind_by_layer, subtask_layer

  def _infer_precondition_by_boolean_reduction(
      self,
      completion_by_subtask: np.ndarray): # [comp_mat, ..] x #subtask, dtype=np.bool
    # 0. prepare
    verbose = False
    union_kvec_by_subtask, sop_by_subtask = [], []
    for subtask_idx in range(self._num_subtask):
      # Use comp_mat as sop. simplify it
      comp_mat = completion_by_subtask[subtask_idx] # [#rows x num_subtask]
      sop = self.simplify_sop(comp_mat)
      if sop.shape[0] == 1:
        union_kvec = sop.squeeze()
      else:
        sop = self.lossy_simplify_sop(sop, max_depth=4, max_num_literal=3, verbose=verbose)
        if len(sop) < len(comp_mat):
          print(f'[subtask{subtask_idx}] #literals={len(comp_mat)} -> {len(sop)}')
        # do OR over all kvec
        union_kvec = np.zeros((self._num_subtask))
        for kvec in sop:
          union_kvec = np.logical_or(union_kvec, kvec)
      # record
      assert union_kvec.ndim == 1
      union_kvec_by_subtask.append(union_kvec)
      sop_by_subtask.append(sop)

    ## 0-1. remove child-of-child
    sop_by_subtask = self.remove_grandchild(sop_by_subtask)

    # 1. Find first layer: is first layer if elig when np.all(comp==False)
    curr_layer_ind_list = []
    for subtask_idx in range(self._num_subtask):
      sop = sop_by_subtask[subtask_idx]
      if sop.shape[0] == 1 and np.all(~sop):
        curr_layer_ind_list.append(subtask_idx)

    ### Init with first layer
    cand_ind_list = curr_layer_ind_list.copy()
    tind_by_layer = [curr_layer_ind_list]
    subtask_layer = np.full((self._num_subtask), fill_value=-1, dtype=np.int16)
    subtask_layer[np.array(curr_layer_ind_list)] = 0 # assign first layer

    # 2. Try with strict criteria
    for layer_ind in range(1, self._num_subtask):
      # Inference done.
      if len(cand_ind_list) == self._num_subtask:
        break
      cand_vec = logic_utils.to_multi_hot(np.array(cand_ind_list), self._num_subtask)
      curr_layer_ind_list = []
      for subtask_idx in range(self._num_subtask):
        if subtask_layer[subtask_idx] >= 0: # skip already assigned subtask
          continue
        union_kvec = union_kvec_by_subtask[subtask_idx]
        if logic_utils.a_in_b(a=union_kvec, b=cand_vec):
          # if simplified expression only contains cand_ind_list, then add it.
          curr_layer_ind_list.append(subtask_idx)
          subtask_layer[subtask_idx] = layer_ind

      if len(curr_layer_ind_list) == 0:
        print('Warning!!!!! Layer-wise ILP failed')
        return sop_by_subtask, None, subtask_layer
      assert len(curr_layer_ind_list) > 0, f"Error: layer-wise ILP failed to find precondition for subtasks {np.nonzero(subtask_layer==-1)}"
      cand_ind_list.extend(curr_layer_ind_list)
      tind_by_layer.append(curr_layer_ind_list)

    return sop_by_subtask, tind_by_layer, subtask_layer

  def remove_grandchild(self, sop_by_subtask):
    # 1. for each subtask, fill-out direct child.
    dc_mask_by_subtask = []
    for sop in sop_by_subtask:
      count = sop.sum(axis=0)
      direct_child_mask = count == len(sop)
      dc_mask_by_subtask.append(direct_child_mask)

    # 2. For each subtask iterate over each child. check if 1) it has direct child, 2) the direct child is also a child of grandparent. if yes, remove grandchild from grand parent.
    for sub_idx, sop in enumerate(sop_by_subtask):
      for row_idx, kvec in enumerate(sop):
        idxs = np.argwhere(kvec)[:, 0]
        assert idxs.ndim == 1
        for idx in idxs: # iterate over child
          # get the child(=idx)'s direct child
          grandchildren_mask = dc_mask_by_subtask[idx]
          multiplier = ~grandchildren_mask
          """if (grandchildren_mask * kvec).sum() > 0:
            import ipdb; ipdb.set_trace()"""
          sop_by_subtask[sub_idx][row_idx] = sop_by_subtask[sub_idx][row_idx] * multiplier
          #kvec = kvec * multiplier # 1->0 all the grandchildren
    return sop_by_subtask

  def lossy_simplify_sop(self, sop, max_depth=3, max_num_literal=3, num_ones_threshold=5, verbose=False):
    num_literal, num_subtask = sop.shape
    num_ones = sop.sum()
    if num_literal <= max_num_literal and num_ones <= num_ones_threshold:
      return sop

    best_sop = None
    depth_threshold = max_depth
    while best_sop is None:
      if num_ones <= num_ones_threshold*2 and num_literal <= max_num_literal*2:
        best_sop = self.simplify_sop_by_element(sop, depth_threshold, max_num_literal, num_ones_threshold, True, verbose)
        if best_sop is None:
          best_sop = self.simplify_sop_by_element(sop, depth_threshold, max_num_literal, num_ones_threshold, False, verbose)
      else:
        best_sop = self.simplify_sop_by_column(sop, depth_threshold, max_num_literal, num_ones_threshold, True, verbose)
        if best_sop is None:
          best_sop = self.simplify_sop_by_column(sop, depth_threshold, max_num_literal, num_ones_threshold, False, verbose)
      if best_sop is None:
        print(f'Failed with depth={depth_threshold}...retrying with depth={depth_threshold+1}')
      depth_threshold = depth_threshold + 1
    return best_sop

  def simplify_sop_by_element(self, sop, max_depth=3, max_num_literal=3, num_ones_threshold=5, protect_all_ones=False, verbose=False):
    # init
    num_literal, num_subtask = sop.shape
    num_ones = sop.sum()
    if max_depth >= num_ones:
      return sop

    # get candiates
    one_counts = sop.sum(axis=0)
    candidates = np.argwhere(sop)
    num_candidates = len(candidates)
    mask = np.ones((num_candidates), dtype=bool)
    for idx in range(num_candidates):
      row, col = candidates[idx]
      if sop[row, :].sum() == 1: # the only 1 in the row, then removing it will results in all-zero.
        mask[idx] = 0
      if protect_all_ones and one_counts[col] == num_literal: # if the column is all-ones, protect it (no mutation)
        mask[idx] = 0
    candidates = candidates[mask] # skip the candidates that results in all zero
    if len(candidates) == 0:
      return None # failure
    num_candidates = len(candidates)
    #
    root = SearchNode(sop=sop, depth=0, choice_vec=np.zeros((num_candidates), dtype=bool))
    job_stack = queue.SimpleQueue()
    job_stack.put(root)
    min_loss = -1
    ever_always_elig = False
    choice_code_set = set()
    while not job_stack.empty():
      node = job_stack.get()
      sop = node.sop
      depth = node.depth
      choice_vec = node.choice_vec

      for idx in range(num_candidates):
        row, col = candidates[idx]
        new_choice_vec = choice_vec.copy()
        new_choice_vec[idx] = True
        new_sop = sop.copy()
        new_sop[row, col] = False
        #
        simple_new_sop = self.simplify_sop(new_sop)
        num_ones = simple_new_sop.sum()
        num_literal = simple_new_sop.shape[0]
        if num_ones == 0:
          # reduced too much
          ever_always_elig = True
          continue
        loss = pow(1 + depth + 1, 0.5) * pow(num_literal, 3) * pow(num_ones, 2)
        assert loss > 0
        if num_literal <= max_num_literal and verbose:
          print(f'[*] loss={loss}, #literals={num_literal}')
        #else:
        #  print(f'loss={loss}, #literals={num_literal}')

        # compute loss and update best
        if num_literal <= max_num_literal and (min_loss > loss or min_loss == -1):
          min_loss = loss
          best_sop = simple_new_sop
          best_choice_vec = choice_vec
          best_num_ones = num_ones

        # branch children
        if depth < max_depth and (num_literal > max_num_literal or num_ones > num_ones_threshold):
          new_choice_code = logic_utils.batch_bin_encode(new_choice_vec)
          if new_choice_code not in choice_code_set: # avoid duplication
            choice_code_set.add(new_choice_code)
            new_node = SearchNode(new_sop, depth+1, new_choice_vec)
            job_stack.put(new_node)

    # return best outcome
    if min_loss == -1:
      if ever_always_elig:
        return np.zeros((1, num_subtask), dtype=bool)
      else:
        return None
    else:
      #print(f'best: loss={min_loss}, num_child={len(best_sop)}, num_ones={best_num_ones}, choice={best_choice_vec.astype(int)}')
      return best_sop

  def simplify_sop_by_column(self, sop, max_depth=3, max_num_literal=3, num_ones_threshold=5, protect_all_ones=False, verbose=False):
    # init
    num_literal, num_subtask = sop.shape
    if max_depth >= num_subtask:
      return sop
    root = SearchNode(sop=sop, depth=0, choice_vec=np.zeros((num_subtask), dtype=bool), num_edit=0)
    job_stack = queue.SimpleQueue()
    job_stack.put(root)
    min_loss = -1
    choice_code_set = set()
    while not job_stack.empty():
      node = job_stack.get()
      sop = node.sop
      num_edit = node.num_edit
      depth = node.depth
      choice_vec = node.choice_vec
      positive_counts = sop.sum(axis=0)
      for subtask_idx in range(self._num_subtask):
        n_edit = positive_counts[subtask_idx]
        if protect_all_ones and n_edit == num_literal:
          continue
        new_choice_vec = choice_vec.copy()
        new_choice_vec[subtask_idx] = True
        if n_edit == 0:
          continue
        new_sop = sop.copy()
        new_sop[:, subtask_idx] = False
        #
        simple_new_sop = self.simplify_sop(new_sop)
        num_ones = simple_new_sop.sum()
        num_literal = simple_new_sop.shape[0]
        if num_ones == 0:
          # reduced too much
          continue
        loss = pow(1 + num_edit + n_edit, 0.5) * pow(num_literal, 3) * pow(num_ones, 2)
        assert loss > 0
        if num_literal <= max_num_literal and verbose:
          print(f'[*] loss={loss}, #literals={num_literal}')
        #else:
        #  print(f'loss={loss}, #literals={num_literal}')

        # compute loss and update best
        if num_literal <= max_num_literal and (min_loss > loss or min_loss == -1):
          min_loss = loss
          best_sop = simple_new_sop
          best_choice_vec = choice_vec
          best_num_ones = num_ones
          best_num_edit = num_edit + n_edit

        # branch children
        if depth < max_depth and (num_literal > max_num_literal or num_ones > num_ones_threshold):
          new_choice_code = logic_utils.batch_bin_encode(new_choice_vec)
          if new_choice_code not in choice_code_set: # avoid duplication
            choice_code_set.add(new_choice_code)
            new_node = SearchNode(new_sop, depth+1, new_choice_vec, num_edit+n_edit)
            job_stack.put(new_node)
        #

    # return best outcome
    if min_loss == -1:
      return None
    else:
      #print(f'best: loss={min_loss}, num_edit={best_num_edit}, num_child={len(best_sop)}, num_ones={best_num_ones}, choice={best_choice_vec.astype(int)}')
      return best_sop

  def _check_validity(self, inputs, targets):
    # inputs: np.arr((T, ntasks), dtype=bool) (only 0/1)
    # targets: np.arr((T,), dtype=bool) (only 0/1)
    assert inputs.dtype == np.bool and targets.dtype == np.bool, "type error"
    assert inputs.ndim == 2 and targets.ndim == 1, "shape errror"

    # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
    # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
    code_batch = np.asarray(logic_utils.batch_bin_encode(inputs))
    eligible_code_set = set(code_batch[targets])
    ineligible_code_set = set(code_batch[np.logical_not(targets)])
    return eligible_code_set.isdisjoint(ineligible_code_set)

  def simplify_sop(self, sop_tensor):
    """
      This function performs the following two reductions
      A + AB  -> A
      A + A'B -> A + B

      sop_bin: binarized sop. (i.e., +1 -> +1, 0 -> 0, -1 -> +1)
    """
    numAND = sop_tensor.shape[0]
    mask = np.ones(numAND, dtype=bool)
    max_iter = 20

    for jj in range(max_iter):
      done = True
      remove_list = []
      sop_bin = np.not_equal(sop_tensor,0).astype(np.uint8)

      for i in range(numAND):
        if mask[i] == 1:
          kbin_i = sop_bin[i]

          for j in range(i + 1, numAND):
            if mask[j] == 1:
              kbin_j = sop_bin[j]
              kbin_mul = kbin_i * kbin_j

              if np.all(kbin_mul == kbin_i):  # i subsumes j. Either 1) remove j or 2) reduce j.
                done = False
                sop_common_j = sop_tensor[j] * kbin_i  # common parts in j.
                difference_tensor = sop_common_j != sop_tensor[i]  # (A,~B)!=(A,B) -> 'B'
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes --> remove j.
                  mask[j] = 0
                else:  # turn off the different bits
                  dim_ind = np.nonzero(difference_tensor)[0]
                  sop_tensor[j][dim_ind] = 0

              elif np.all(kbin_mul == kbin_j):  # j subsumes i. Either 1) remove i or 2) reduce i.
                done = False
                sop_common_i = sop_tensor[i] * kbin_j
                difference_tensor = sop_common_i != sop_tensor[j]
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes--> remove i.
                  mask[i] = 0
                else:  # turn off the different bit.
                  dim_ind = np.nonzero(difference_tensor)[0]
                  sop_tensor[i][dim_ind] = 0

      if done:
        break

    if mask.sum() < numAND:
      sop_tensor = sop_tensor[mask.nonzero()[0],:]
      #sop_tensor = sop_tensor.index_select(0,mask.nonzero().view(-1))

    return sop_tensor, mask

class SearchNode:
  def __init__(self, sop, depth, choice_vec, num_edit=None) -> None:
    self.sop = sop
    self.num_edit = num_edit
    self.depth = depth
    self.choice_vec = choice_vec
