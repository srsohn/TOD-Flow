import numpy as np
from sklearn.tree import DecisionTreeClassifier
from util.graph_utils import _sktree_to_sop, dotdict
from util import logic_utils

class BCILP: # behavioral-cloning ILP
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
    
    # Decision tree
    self.min_leaf_frac = ilp_args['min_leaf_frac']
    self.max_depth = ilp_args['max_depth']
    self.max_features = ilp_args['max_features'] if ilp_args['max_features'] > 0 else None # max # of features used in DT
    
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
    return f"BCILP"

  def infer_task(self):
    sop_set = []
    for op in range(self.num_option):
      # initialize
      mask = np.ones(self.num_subtask, dtype=bool)
      comp = self.comp_by_op[op]
      elig = self.elig_by_op[op]
      weights = self.weights_by_op[op]
      # run ILP
      sop_tensor = self._infer_op(comp, elig, mask=mask, weights=weights, op_idx=op)
      sop_set.append(sop_tensor)
    return sop_set
  
  def _infer_op(self, comp, elig, weights=None, mask=None, op_idx=None):
    if comp.shape[0] == 0:
      print('Warning: no data left for ILP')
      return None
    if mask is None:
      mask = np.ones(comp.shape[1], dtype=np.bool_)
    if weights is None:
      weights = np.ones(elig.shape, dtype=np.float_)
      weights /= weights.sum()

    #assert np.abs(weights.sum() - 1) < 1e-3
    assert mask.shape[0] == comp.shape[1]
    assert elig.dtype == np.bool_

    comp = comp[:, mask]
    dt = DecisionTreeClassifier(
        max_depth=self.max_depth,
        min_weight_fraction_leaf=self.min_leaf_frac,
        max_features=self.max_features,
        random_state=0,
    )
    dt.fit(comp, elig, weights)
    m_sop = _sktree_to_sop(dt, comp.shape[1])
    if m_sop is None:
      return None
    m_sop_tensor, _ = self.simplify_sop(m_sop)
    assert m_sop_tensor.ndim == 2 or m_sop_tensor.shape == (1, ), "sop_tensor should be 2 dimension or scalar"
    sop_tensor = np.zeros((m_sop_tensor.shape[0], mask.shape[0]), dtype=m_sop_tensor.dtype)
    sop_tensor[:, mask] = m_sop_tensor
    if (sop_tensor == 0).all():
      return None
    return sop_tensor

  
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

    self._data_by_option(self.completion, self.options)

  def _data_by_option(self, comp, ops):
    self.comp_by_op, self.elig_by_op, self.weights_by_op = {}, {}, {}
    assert len(comp) == len(ops)
      
    for op_idx in range(self.num_option):
      self.comp_by_op[op_idx] = []
      self.elig_by_op[op_idx] = []
      self.weights_by_op[op_idx] = []
      for traj_comp, traj_ops in zip(comp, ops):
        assert traj_comp.shape[0] == traj_ops.shape[0]
        # 1. should label
        should_labels = traj_ops[:, op_idx]
        
        # 2. weight = 1.0 everywhere
        should_weight = np.ones_like(should_labels).astype(float)
        
        # 3. completion label
        prev_comp = np.zeros_like(traj_comp)
        prev_comp[1:] = traj_comp[:-1]
        comp_label = prev_comp
            
        self.comp_by_op[op_idx].append(comp_label)
        self.elig_by_op[op_idx].append(should_labels)
        self.weights_by_op[op_idx].append(should_weight)

      if len(self.comp_by_op[op_idx]) > 0:
        self.comp_by_op[op_idx] = np.concatenate(self.comp_by_op[op_idx], axis=0)
        self.elig_by_op[op_idx] = np.concatenate(self.elig_by_op[op_idx], axis=0)
        self.weights_by_op[op_idx] = np.concatenate(self.weights_by_op[op_idx], axis=0)
        #self.weights_by_op[op_idx] /= self.weights_by_op[op_idx].sum()
      else:
        # edge case, option never appears (default to always eligible)
        self.comp_by_op[op_idx] = np.zeros((1, self.num_subtask), dtype=np.bool_)
        self.elig_by_op[op_idx] = np.ones((1,), dtype=np.bool_)
        self.weights_by_op[op_idx] = np.ones((1,), dtype=np.float_)
  
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