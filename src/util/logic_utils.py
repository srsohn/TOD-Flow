import numpy as np
from pyeda.inter import expr, exprvar

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

def a_in_b(a, b):
  # a: [1, 1, 0 0], b: [1,1,0,1] --> True
  assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
  assert a.dtype == bool and b.dtype == bool

  if a.ndim==1 and b.ndim==1:
    assert len(a) == len(b)
    a_or_b = np.logical_or(a, b)
    return np.all(b == a_or_b)
  
  else:
    # TODO: implement
    return 

def multihot_to_indices(multihot_tensor):
  return np.argwhere(multihot_tensor)

def to_multi_hot(index_tensor, max_dim):
  # number-to-onehot or numbers-to-multihot
  if isinstance(index_tensor, list):
    index_tensor = np.array(index_tensor)
  if index_tensor.ndim == 1:
    if index_tensor.shape[0] == 0: # empty
      out = np.zeros((max_dim), dtype=bool)
    else:
      out = (np.expand_dims(index_tensor, axis=1) == \
            np.arange(max_dim).reshape(1, max_dim))
      assert out.ndim == 2
      out = out.sum(axis=0).astype(bool)
  else:
    out = (index_tensor == np.arange(max_dim).reshape(1, max_dim))
  assert isinstance(out, np.ndarray), "output should be np.ndarray"
  return out

def batch_bin_encode(bin_tensor):
  if isinstance(bin_tensor, list) or bin_tensor.ndim == 2:
    return [hash(row.tobytes()) for row in bin_tensor]
  elif bin_tensor.ndim==1:
    return hash(bin_tensor.tobytes())

def and_expr_to_vec(and_expr, num_feat):
  andvec = np.zeros((num_feat), dtype=int)
  if 'And' in str(and_expr):
    feat_list = and_expr.xs
    for feat_expr in feat_list:
      feat_str = str(feat_expr)
      value = 1 if feat_str[0] == '~' else -1
      index = int(feat_str[feat_str.index('_')+1:])
      andvec[index] = value
  else:
    feat_str = str(and_expr)
    value = 1 if feat_str[0] == '~' else -1
    index = int(feat_str[feat_str.index('_')+1:])
    andvec[index] = value
  return andvec

def expr_to_graph(pcond_list, num_subtask):
  sop_set = []
  for pcond in pcond_list:
    pcond_str = str(pcond)
    andvec_list = []
    if 'Or' in pcond_str:
      and_expr_list = pcond.xs
      for and_expr in and_expr_list:
        andvec = and_expr_to_vec(and_expr, num_subtask)
        andvec_list.append(andvec)
      sop_mat = np.stack(andvec_list)
    elif isinstance(pcond, int): # True, False
      if pcond == 1:
        sop_mat = np.zeros((1, num_subtask), dtype=int)
      else:
        sop_mat = np.zeros((1, num_subtask), dtype=int)
    else:
      and_expr = pcond
      andvec = and_expr_to_vec(and_expr, num_subtask)
      sop_mat = np.expand_dims(andvec, 0)
    assert sop_mat.ndim == 2
    sop_set.append(sop_mat)
  return sop_set
  #return sop_set, tind_by_layer, subtask_layer
        