import numpy as np
from util.graph_utils import _sktree_to_sop, dotdict
from util import logic_utils

class SHDILP:
  def __init__(
      self,
      ilp_args: dict,
      # Dataset
      task_name: str = None,
      subtask_label: list = None,
      option_label: list = None,
      # 
      verbose: bool = False):

    # Beam
    self.score_bias = 1.0
    self.minimum_score = ilp_args['min_score']
    self.beam_width = ilp_args['beam_width']
    self.beam_depth = ilp_args['beam_depth']
    self.complexity_penalty = ilp_args['complexity_penalty']
    
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
    return f"SHDILP"

  def infer_task(self):
    sop_set = []
    for op in range(self.num_option):
      # initialize
      option_vec = self.entire_option_matrix[:, op]
      comp_matrix = self.entire_prev_comp_matrix.copy()
      if option_vec.sum() == 0:
        sop_set.append(False)
        continue
      # beam search for finding best and_vec (& preconditions)
      if False:
        print(f'Option {op}: {self.option_labels[op]}')
        
      sop_tensor = self._beam_search_precondition(option_vec, comp_matrix)
      if (sop_tensor == 0).all():
        sop_tensor = None
      sop_set.append(sop_tensor)
    return sop_set
  
  def _expand_hypothesis(self, prev_sop_tensor, cache):
    # prev_matrix  #[B x N_and x N_subtask]
    prev_sop_tensor_=prev_sop_tensor.copy()
    assert prev_sop_tensor.ndim == 3
    num_subtask = prev_sop_tensor.shape[-1]
    hypothesis_sop_list = []
    for prev_sop_mat in prev_sop_tensor:
      # 1. expand hypothesis
      # 1-1. Assign one element to the place where it's 0
      xs, ys = (prev_sop_mat==0).nonzero()
      for x, y in zip(xs, ys):
        prev_sop_mat[x, y] = 1
        key = tuple(prev_sop_mat.flatten().tolist())
        if key not in cache:
          hypothesis_sop_list.append(prev_sop_mat.copy())
          cache.add(key)
        
        prev_sop_mat[x, y] = -1
        key = tuple(prev_sop_mat.flatten().tolist())
        if key not in cache:
          hypothesis_sop_list.append(prev_sop_mat.copy())
          cache.add(key)
        # revert to original value
        prev_sop_mat[x, y] = 0
      
      # 1-2. Assign one column with +/-1
      for i in range(num_subtask):
        new_sop_mat = prev_sop_mat.copy()
        new_sop_mat[:, i] = 1
        key = tuple(new_sop_mat.flatten().tolist())
        if key not in cache:
          hypothesis_sop_list.append(new_sop_mat)
          cache.add(key)
        
      # 1-3. Add one row with one element +1/-1
      is_row_all_ones = np.all(prev_sop_mat==1, axis=1)
      if len(is_row_all_ones.nonzero()[0]) > 0:
        next_row_index = is_row_all_ones.nonzero()[0][0]
        prev_sop_mat[next_row_index, :] = 0
        for i in range(num_subtask):
          prev_sop_mat[next_row_index, i] = 1
          key = tuple(prev_sop_mat.flatten().tolist())
          if key not in cache:
            hypothesis_sop_list.append(prev_sop_mat.copy())
            cache.add(key)
          
          prev_sop_mat[next_row_index, i] = -1
          key = tuple(prev_sop_mat.flatten().tolist())
          if key not in cache:
            hypothesis_sop_list.append(prev_sop_mat.copy())
            cache.add(key)
          prev_sop_mat[next_row_index, i] = 0
        # revert back to all one (i.e., all-one row means not opened yet.)
        prev_sop_mat[next_row_index, :] = 1
    
    if len(hypothesis_sop_list) == 0: # possible when top5 stays the same as previous step.
      return None
    hypothesis_sop_tensor = np.stack(hypothesis_sop_list, axis=0) # stack([2d mats]) -> 3d mat
    assert hypothesis_sop_tensor.ndim == 3
    assert np.all(prev_sop_tensor_ == prev_sop_tensor)
    hypothesis_sop_tensor = np.concatenate([prev_sop_tensor, hypothesis_sop_tensor], axis=0) # concat([3d mats]) -> 3d mat
    assert hypothesis_sop_tensor.ndim == 3
    return hypothesis_sop_tensor #[num_hyp x N_and x N_subtask]
      
  def _beam_search_precondition(self, option_vec, comp_matrix):
    # option_vec  : [total_steps]
    # comp_matrix : [total_steps x num_subtask]
    num_subtask = comp_matrix.shape[1]
    total_steps = option_vec.shape[0]
    comp_pm_matrix = comp_matrix.astype(int) * 2 - 1
    assert option_vec.ndim == 1
    assert comp_matrix.shape[0] == total_steps and comp_matrix.shape[1] == self.num_subtask, f"{comp_matrix.shape[0]} != {total_steps} or {comp_matrix.shape[1]} != {self.num_subtask}"
    num_max_and = self.beam_depth
    prev_best_hypothesis_sop_tensor = np.ones( (1, num_max_and, self.num_subtask), dtype=int)
    prev_best_hypothesis_sop_tensor[0,0,:] = 0
    
    num_option_executed = int(option_vec.sum())
    assert num_option_executed > 0, "Error"
    cache = set()
    for depth in range(self.beam_depth):
      # 1. List up new hypotheses by expanding one literal
      hypothesis_sop_tensor = self._expand_hypothesis(prev_best_hypothesis_sop_tensor, cache)
      if hypothesis_sop_tensor is None: # cannot find better hypothesis! stop beam search.
        break
      assert hypothesis_sop_tensor.ndim == 3
      # hypothesis_sop_tensor: [num_hyp x N_max_and x N_subtask]
      # 2. Compute score
      hypothesis_sop_tensor_bias = np.count_nonzero(hypothesis_sop_tensor, axis=2) # [num_hyp x N_max_and]
      skip_mask = hypothesis_sop_tensor_bias == num_subtask
      hypothesis_sop_tensor_bias_masked = hypothesis_sop_tensor_bias.copy()
      hypothesis_sop_tensor_bias_masked[skip_mask] = 0
      hypothesis_complexity = hypothesis_sop_tensor_bias_masked.sum(axis=1) - 1 # [num_hyp]
      
      # comp_pm_matrix: [num_data x N_subtask] = 982 x 38
      # hypothesis_sop_tensor: [num_hyp x N_max_and x N_subtask] = 191 x 2 x 38
      # [num_data x N_subtask] \dot [num_hyp x N_max_and x N_subtask] = [num_data, num_hyp x N_max_and ] = 982 x 191 x 2
      matmul_res = np.tensordot(comp_pm_matrix, hypothesis_sop_tensor,axes=((1),(2)))
      #matmul_res = comp_pm_matrix @ hypothesis_sop_tensor.T # [num_data x num_hyp x N_max_and]
      hypothesis_sop_tensor_bias_expanded = np.expand_dims(hypothesis_sop_tensor_bias, axis=0) # [1 x num_hyp x N_max_and]
      is_and_cond_satisfied = matmul_res == hypothesis_sop_tensor_bias_expanded # [num_data x num_hyp x N_max_and]
      # [num_data x num_hyp x N_max_and] -> [num_data x num_hyp]
      num_and_cond_satisfied = is_and_cond_satisfied.sum(axis=2)
      # if any AND condition is satisfied, then hypothesis is True
      hypothesis_true_matrix = num_and_cond_satisfied > 0 # [num_data x num_hyp]
      #
      num_hypothesis_true = hypothesis_true_matrix.sum(axis=0, keepdims=True) # [1 x (num_hyp)]
      #hypothesis_false_when_option = option_vec.astype(int) @ hypothesis_false_matrix # [1 x T] @ [T x (num_hyp)] = [1 x (num_hyp)]
      num_hypothesis_true_when_option = option_vec.astype(int) @ hypothesis_true_matrix # [1 x T] @ [T x (num_hyp)] = [1 x (num_hyp)]
      
      bias = self.score_bias
      score = (num_hypothesis_true_when_option) / (bias + num_hypothesis_true) - self.complexity_penalty * hypothesis_complexity # higher better
      
      # 3. Choose Top-B hypotheses
      best_inds = (-score.flatten()).argsort()[:self.beam_width]
      prev_best_hypothesis_sop_tensor = hypothesis_sop_tensor[best_inds, :, :] # [beam_widht x N_max_and x N_subtask]
    best_score = score.max()
    if best_score > self.minimum_score:
      best_hypothesis_sop_matrix = prev_best_hypothesis_sop_tensor[0] # [N_max_and x N_subtask]
    else:
      best_hypothesis_sop_matrix = np.zeros( (1, self.num_subtask), dtype=int)
    del cache
    if False:
      best_index = best_inds[0]
      print(f'num_preconditions={np.count_nonzero(best_hypothesis_sop_matrix)}')
      for i, val in enumerate(best_hypothesis_sop_matrix):
        if val > 1e-3 or val < -1e-3:
          prefix="Positive" if val > 0 else "Negative"
          print(f'- {prefix} preconditions: [{i}] {self.subtask_labels[i]}, ')
      print(f"score: {masked_score[0, best_index]} violation: {ratio_violation_over_entire_option[best_index]} action reduction: {ratio_successful_action_space_reduction[0, best_index]}")
    keep_row_mask = ~np.all(best_hypothesis_sop_matrix, axis=1)
    best_hypothesis_sop_matrix = best_hypothesis_sop_matrix[keep_row_mask]
    return best_hypothesis_sop_matrix

  
  ############ Data treatment
  def add_data(self, so_ilp_data, soft_effect_mat):
    self.so_ilp_data = so_ilp_data
    self.num_subtask = so_ilp_data['num_subtask']
    self.num_option = so_ilp_data['num_option']
    self.subtask_labels = np.array(so_ilp_data['subtask_labels'])
    self.option_labels = np.array(so_ilp_data['option_labels'])
    self.completion = so_ilp_data['completion'] # num_traj x [np.arr(traj_len, num_option, dtype=bool)]
    self.options = so_ilp_data['options']       # num_traj x [np.arr(traj_len, num_subtask, dtype=bool)]
    self.entire_option_matrix = np.concatenate(self.options, axis=0)        # total_len x num_option, dtype=bool
    #self.entire_completion_matrix = np.concatenate(self.completion, axis=0) # total_len x num_subtask, dtype=bool
    prev_comp_matrix_list = []
    for comp_matrix in self.completion:
      prev_comp_matrix = np.zeros_like(comp_matrix)
      prev_comp_matrix[1:, :] = comp_matrix[:-1, :]
      prev_comp_matrix_list.append(prev_comp_matrix)
    self.entire_prev_comp_matrix = np.concatenate(prev_comp_matrix_list, axis=0)
    
    self.effect_mat = soft_effect_mat
    self.any_effect_mat = (self.effect_mat > 1e-3).any(-1)