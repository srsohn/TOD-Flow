import numpy as np
from util.graph_utils import EffectGraphVisualizer

class EffectILP(object):
  @property
  def algo_name(self):
    return f"EffectILP"

  def add_data(self, so_ilp_data):
    self.so_ilp_data = so_ilp_data
    self.num_subtask = so_ilp_data['num_subtask']
    self.num_option = so_ilp_data['num_option']
    self.subtask_labels = np.array(so_ilp_data['subtask_labels'])
    self.option_labels = np.array(so_ilp_data['option_labels'])
    self.completion = so_ilp_data['completion']
    self.options = so_ilp_data['options']

    self.delta_comp = []
    for comp in self.completion:
      c = np.zeros((comp.shape[0]+1, comp.shape[1]), dtype=comp.dtype)
      c[1:] = comp
      delta = c[1:].astype(np.int_) - c[:-1].astype(np.int_)
      self.delta_comp.append(delta)

    self.X = np.concatenate(self.options)
    self.Y = np.concatenate(self.delta_comp)
    self.Y_comp = np.concatenate(self.completion)
    assert self.X.shape[0] == self.Y.shape[0]
    assert self.Y_comp.shape == self.Y.shape

  def _debug_hard_mat(self, effect_mat):
    for op in range(self.num_option):
      print(self.option_labels[op], '-> +', list(self.subtask_labels[effect_mat[op] == 1]), ', -', list(self.subtask_labels[effect_mat[op] == -1]))

  def _debug_soft_mat(self, effect_mat):
    for op in range(self.num_option):
      print(
          self.option_labels[op],
          '-> +', list(self.subtask_labels[(0 < effect_mat[op, :, 1]) & (effect_mat[op, :, 1] < 1)]),
          ', -', list(self.subtask_labels[(0 < effect_mat[op, :, 0]) & (effect_mat[op, :, 0] < 1)])
      )

  def visualize(self, vis_filename):
    visualizer = EffectGraphVisualizer()
    dot = visualizer.visualize_soft(self.soft_effect_mat, self.subtask_labels, self.option_labels)
    visualizer.render_and_save(dot, vis_filename)

  def infer_task(self):
    explained = self.Y == 0

    # infer hard effects first
    ## Filter by options->subtask where subtask is always postive / negative
    pure_pos_effect = np.zeros((self.num_option, self.num_subtask), dtype=np.bool_)
    pure_neg_effect = np.zeros((self.num_option, self.num_subtask), dtype=np.bool_)
    for option in range(self.num_option):
      pure_pos_effect[option] = self.Y_comp[self.X[:, option]].all(0)
      pure_neg_effect[option] = (~self.Y_comp[self.X[:, option]]).all(0)

    ## Calculate hard effects
    hard_effect_mat = np.zeros((self.num_option, self.num_subtask), dtype=np.int_)
    while True:
      scores = np.zeros((self.num_option, self.num_subtask), dtype=np.int_)
      for option in range(self.num_option):
        hard_positives = ((self.Y[self.X[:, option]] == 1) & ~explained[self.X[:, option]]).sum(0) * pure_pos_effect[option]
        hard_negatives = ((self.Y[self.X[:, option]] == -1) & ~explained[self.X[:, option]]).sum(0) * pure_neg_effect[option]
        scores[option] = hard_positives + hard_negatives

      best_op, best_st = np.unravel_index(scores.argmax(), scores.shape)
      best_score = scores[best_op, best_st]
      if best_score == 0:
        break

      assert pure_pos_effect[best_op, best_st] | pure_neg_effect[best_op, best_st]
      if pure_pos_effect[best_op, best_st]:
        hard_effect_mat[best_op, best_st] = 1
        assert (self.Y_comp[self.X[:, best_op], best_st]).all()
      else:
        hard_effect_mat[best_op, best_st] = -1
        assert (~self.Y_comp[self.X[:, best_op], best_st]).all()

      explained[self.X[:, best_op], best_st] = True

    # self._debug_hard_mat(hard_effect_mat)

    # infer soft effects next.
    # soft_effect_mat = hard_effect_mat.astype(np.float32)
    soft_effect_mat = np.zeros((self.num_option, self.num_subtask, 2), dtype=np.float32)
    soft_effect_mat[hard_effect_mat == 1, 1] = 1
    soft_effect_mat[hard_effect_mat == -1, 0] = 1
    pos_counts = np.zeros((self.num_option, self.num_subtask), dtype=np.int_)
    neg_counts = np.zeros((self.num_option, self.num_subtask), dtype=np.int_)
    for option in range(self.num_option):
      pos_counts[option] = self.Y_comp[self.X[:, option]].sum(0)
      neg_counts[option] = (~self.Y_comp[self.X[:, option]]).sum(0)

    while True:
      non_explained = (~explained).any(1)
      if non_explained.sum() == 0:
        break
      scores = np.full((2, self.num_option, self.num_subtask), -np.inf, dtype=np.float32)

      cache = {}
      for option in range(self.num_option):
        delta_pos = ((self.Y[self.X[:, option]] == 1) & ~explained[self.X[:, option]]).sum(0)
        delta_neg = ((self.Y[self.X[:, option]] == -1) & ~explained[self.X[:, option]]).sum(0)
        delta_pmu = (delta_pos + 1e-3) / (delta_pos + neg_counts[option] + 2e-3)
        delta_nmu = (delta_neg + 1e-3) / (delta_neg + pos_counts[option] + 2e-3)
        cache[option] = (delta_nmu, delta_pmu)

        scores[1, option] = delta_pos * np.log(delta_pmu) + neg_counts[option] * np.log(1 - delta_pmu)
        scores[0, option] = delta_neg * np.log(delta_nmu) + pos_counts[option] * np.log(1 - delta_nmu)
        scores[1, option, delta_pos == 0] = -np.inf
        scores[0, option, delta_neg == 0] = -np.inf

      is_pos, best_op, best_st = np.unravel_index(scores.argmax(), scores.shape)
      soft_effect_mat[best_op, best_st, is_pos] = cache[best_op][is_pos][best_st]
      explained[self.X[:, best_op] & (self.Y[:, best_st] == (1 if is_pos else -1))] = True

      non_explained = (~explained).any(1)

    # self._debug_hard_mat(hard_effect_mat)
    # self._debug_soft_mat(soft_effect_mat)
    self.hard_effect_mat = hard_effect_mat
    self.soft_effect_mat = soft_effect_mat
    return hard_effect_mat, soft_effect_mat
