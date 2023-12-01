"""Graph Reward Propagation (GRProp) policy implementation."""

from typing import Dict, Sequence

import numpy as np
from acme import specs

import tf_utils
from util.graph_utils import SubtaskGraph

import tensorflow as tf
from tensorflow_probability import distributions as tfd


class GRProp:

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      # TODO: Make as optional.
      temp: float = None,
      w_a: float = None,
      beta_a: float = None,
      ep_or: float = None,
      temp_or: float = None,
      verbose: bool = False):
    # Set additional attributes.
    self._action_spec = environment_spec.actions
    self.max_task = self._action_spec.num_values

    self._verbose = verbose
    self._policy_mode = 'RProp'
    self._graph_initialized = False

    self._temp = temp or 200
    self._w_a = w_a or 3.0
    self._beta_a = beta_a or 8.0
    self._ep_or = ep_or or 0.8
    self._temp_or = temp_or or 2.0

    self.compute_RProp_grad_tf = tf.function(self.compute_RProp_grad)

  @property
  def is_ready(self):
    return self._graph_initialized

  def init_graph(self, graphs: Sequence[SubtaskGraph]):
    """Initialize Wa_tensor, Wo_tensor, rew_tensor."""
    assert isinstance(graphs, list), "Loading graph from a file is not supported."

    batch_size = len(graphs)
    self._graph_initialized = True

    # prepare
    self.num_layers = np.array([len(g.tind_by_layer) for g in graphs])
    self.max_num_layer = max([len(g.tind_by_layer) for g in graphs]) - 1
    max_NA = max([g.ANDmat.shape[0] for g in graphs])  #max total-#-A
    max_NP = max([len(g.subtask_reward) for g in graphs])   #max total-#-P
    self.rew_tensor = np.zeros([batch_size, max_NP], dtype=np.float32)
    for bind, graph in enumerate(graphs):
      self.rew_tensor[bind, :len(graph.subtask_reward)] = graph.subtask_reward

    if self.max_num_layer == 0:
      print('Warning!! flat graph!!!')
      self._policy_mode = 'Greedy'
    else:
      self.Wa_tensor  = np.zeros([self.max_num_layer, batch_size, max_NA, max_NP], dtype=np.float32)
      self.Wa_neg     = np.zeros([batch_size, max_NA, max_NP], dtype=np.float32)
      self.Wo_tensor  = np.zeros([self.max_num_layer, batch_size, max_NP, max_NA], dtype=np.float32)
      self.Pmask      = np.zeros([self.max_num_layer+1, batch_size, max_NP, 1], dtype=np.float32)

      for bind, graph in enumerate(graphs):
        tind_by_layer = graph.tind_by_layer
        num_layer = len(tind_by_layer) - 1
        #
        if isinstance(graph.subtask_reward, list):
          graph.subtask_reward = np.array(graph.subtask_reward)

        if num_layer > 0:
          # W_a_neg
          ANDmat  = graph.ANDmat
          ORmat   = graph.ORmat
          self.Wa_neg[bind, :ANDmat.shape[0], :ANDmat.shape[1]] = (ANDmat < 0).astype(np.float32) # only the negative entries
          abias, tbias = 0, graph.numP[0]
          tind_tensor = np.array(tind_by_layer[0], dtype=np.int32)
          mask = np.zeros(max_NP)
          mask[tind_tensor] = 1
          self.Pmask[0, bind, :] = np.expand_dims(mask, axis=-1)

          for lind in range(num_layer):
            # W_a
            na, _ = graph.W_a[lind].shape
            wa = ANDmat[abias:abias+na, :]
            # output is current layer only.
            self.Wa_tensor[lind, bind, abias:abias+na, :wa.shape[1]] = wa

            # W_o
            tind = tind_by_layer[lind + 1]
            wo = ORmat[:, abias:abias+na]    # numA x numP_cumul
            nt, _ = graph.W_o[lind].shape

            # re-arrange to the original subtask order
            # input (or) is cumulative. output is current layer only.
            self.Wo_tensor[lind, bind, :wo.shape[0], abias:abias+na] = wo
            abias += na

            tind_tensor = np.array(tind, dtype=np.int32)
            mask = np.zeros(max_NP)
            mask[tind_tensor] = 1
            self.Pmask[lind + 1, bind, :] = np.expand_dims(mask, axis=-1)
            tbias += nt

      #print('[GRProp]  rew_tensor=',self.rew_tensor.squeeze())
      # only the positive entries
      self.Wa_tensor = (self.Wa_tensor > 0).astype(np.float32)

  def compute_RProp_grad(self, completion, reward, Wa_tensor, Wa_neg, Wo_tensor, Pmask):
    """Computes the grprop gradient."""
    with tf.GradientTape() as tape:
      tape.watch(completion)
      soft_reward = self.RProp(
        completion=completion,
        reward=reward,
        Wa_tensor=Wa_tensor,
        Wa_neg=Wa_neg,
        Wo_tensor=Wo_tensor,
        Pmask=Pmask,
      )
    return tape.gradient(soft_reward, completion)

  #@tf.function(autograph=True, experimental_relax_shapes=False) # This will retrace when Wa_tensor shape changes (i.e., when self.max_num_layer changes)
  def RProp(self, completion, reward, Wa_tensor, Wa_neg, Wo_tensor, Pmask):
    r"""
      completion: Nb x NP x 1         : completion input. {0,1}
      p: precondition progress [0~1]
      a: output of AND node. I simply added the progress.
      or: output of OR node.
      p = softmax(a).*a (soft version of max function)
      or = max (x, \lambda*p + (1-\lambda)*x). After execution (i.e. x=1), gradient should be blocked. So we use max(x, \cdot)
      a^ = Wa^{+}*or / Na^{+} + 0.01   -       Wa^{-}*or.
          -> in [0~1]. prop to #satisfied precond.       --> If any neg is executed, becomes <0.
      a = max( a^, 0 ) # If any neg is executed, gradient is blocked
      Intuitively,
      p: soft version of max function
      or: \lambda*p + (1-\lambda)*x
      a: prop to portion of satisfied precond
    """

    tf.Assert(self.max_num_layer == tf.shape(Wa_tensor)[0], [0])
    tf.Assert(self.max_num_layer == tf.shape(Wo_tensor)[0], [0])
    tf.Assert(self.max_num_layer+1 == tf.shape(Pmask)[0], [0])
    # 1. forward (45% time)
    or_ = tf.maximum(completion, self._ep_or + (0.99 - self._ep_or) * completion) * Pmask[0]
    A_neg = tf.matmul(Wa_neg, completion)     #(Nb x NA x NP) * (Nb x NP x 1) = (Nb x NA x 1)
    or_list = [or_]
    for lind in range(self.max_num_layer):
      #Init
      wa = Wa_tensor[lind]
      wo = Wo_tensor[lind]
      pmask = Pmask[lind + 1]

      or_concat = tf.concat(or_list, axis=2)

      #AND layer
      a_pos = tf.reduce_sum(tf.matmul(wa, or_concat), axis=-1) / (
          tf.maximum(tf.reduce_sum(wa, axis=-1), 1))
      a_pos = tf.expand_dims(a_pos, axis=-1)  # (Nb x NA x 1)
      a_hat = a_pos - self._w_a * A_neg                             #Nb x Na x 1

      #and_ = nn.Softplus(self._beta_a)(a_hat)
      and_ = tf.nn.softplus(a_hat * self._beta_a) / self._beta_a    #Nb x Na x 1 (element-wise)

      #soft max version2
      num_next_or = wo.shape[1]
      and_rep = tf.tile(tf.transpose(and_, [0, 2, 1]), [1, num_next_or, 1])
      p_next = (tf_utils.masked_softmax(self._temp_or * and_rep, wo) * and_rep)
      p_next = tf.reduce_sum(p_next, axis=-1, keepdims=True)

      or_ = tf.maximum(completion, self._ep_or * p_next + (0.99 - self._ep_or) * completion) * pmask  # Nb x Np_sum x 1
      or_list.append(or_)

    # loss (soft reward)  (should be scalar)
    or_mat = tf.concat(or_list, axis=2)
    soft_reward = tf.matmul(tf.transpose(or_mat, [0, 2, 1]), reward) # (Nb x Nt).*(Nb x Nt) (element-wise multiply)
    soft_reward = tf.reduce_sum(soft_reward)
    return soft_reward

  def get_raw_logits(self, observation: Dict[str, np.ndarray]):
    assert self._graph_initialized, \
      'Graph not initialized. "init_graph" must be called at least once before taking action.'

    mask = observation['mask']
    completion = observation['completion']

    num_subtasks = len(completion[0])
    if np.all(self.rew_tensor == 0) or self._policy_mode == 'Greedy':
      print('Warning!! flat')
      logits = self.rew_tensor
    elif self._policy_mode == "RProp":
      # 1. prepare input
      completion = np.expand_dims(completion, axis=-1)
      reward = np.expand_dims(self.rew_tensor * mask, axis=-1)  # mask reward

      # 2. compute grad (48% time)
      gradients = self.compute_RProp_grad_tf(
          completion=completion,
          reward=reward,
          Wa_tensor=self.Wa_tensor,
          Wa_neg=self.Wa_neg,
          Wo_tensor=self.Wo_tensor,
          Pmask=self.Pmask,
      )   # gradient is w.r.t. completion

      if self._verbose:
        print('gradient=', tf.squeeze(gradients).numpy())
      assert gradients.shape == completion.shape
      logits = self._temp * gradients.numpy()
      logits = logits.squeeze(-1)
      assert logits.shape == completion.shape[:-1]

      # if flat, switch to greedy.
      logits += np.tile(
          (self.num_layers == 1).astype(np.float32).reshape(-1, 1),
          (1, num_subtasks)) * self.rew_tensor
    else:
      raise RuntimeError("unknown policy mode :" + self._policy_mode)

    return logits