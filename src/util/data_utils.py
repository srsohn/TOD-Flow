import numpy as np
import json
from util import logic_utils
DIR_BY_DATASET={
        'SGD': '../datasets/dstc8-schema-guided-dialogue/'
        }

class DataLoader(object):

  def __init__(
      self,
      dataset: str = "SGD",
      task: str = "hotel",
      data_path: str = "trajectories.json",
      score_key: str = "",
      dialog_flow_home_dir = "../",
      split = None,
      ):
    self.name = dataset
    self.task_name = task
    self.filename = data_path
    self.score_key = score_key
    self.split = split
    #
    self.data_by_task = dict()
    # prep
    self.load_segment_data(data_path)

  def load_segment_data(self, data_path):
    print(f'reading data @ {data_path}')
    self.so_data_by_task = json.load(open(data_path))
    for task in self.so_data_by_task:
        self.so_data_by_task[task]['num_subtask'] = len(self.so_data_by_task[task]['subtask_labels'])
        self.so_data_by_task[task]['num_options'] = len(self.so_data_by_task[task]['option_labels'])
        # remove data by split if specified
        if self.split is not None:
            new_trajs = []
            for traj in self.so_data_by_task[task]['trajectories']:
                assert 'split' in traj
                if traj['split'] == self.split:
                    new_trajs.append(traj)
            self.so_data_by_task[task]['trajectories'] = new_trajs
    self.task_names = list(self.so_data_by_task.keys())
    #
    self.num_task = len(self.data_by_task)
    self.data_by_task = self._spoof_video_data(self.so_data_by_task)

  # data_by_task: dict
  # data = data_by_task[task_name]: dict
  # data: dict
  # - num_subtask: int
  # - num_option: int
  # - subtask_labels: list of str, names of subtasks in order
  # - option_labels: list of str, names of options in order
  # - trajectories: [dict, ..]
  # -- trajectory: dict
  # --- name: str
  # --- subtask_and_option_indices: [list]
  # ---- (each timestep) : list of length 2
  # ----- 0: "subtask"/"option"
  # ----- 1: list of subtask/option indices at this timestep

  def _spoof_video_data(self, so_data):
    # convert option-subtask format into subtask-only format used in video ILP
    data_by_task = {}
    for name in so_data: 
        data = {}
        data['subtask_labels'] = so_data[name]['subtask_labels'] + so_data[name]['option_labels']
        data['num_subtask'] = len(so_data[name]['subtask_labels']) + len(so_data[name]['option_labels'])
        data['trajectories'] = []

        for raw_trajectory in so_data[name]['trajectories']:
            N = len(raw_trajectory['subtask_and_option_indices'])
            trajectory = {}
            trajectory['subtask_indices'] = []
            trajectory['start_flags'] = []
            seen = set()
            for i, (turn_type, idxs) in enumerate(raw_trajectory['subtask_and_option_indices']):
                if turn_type == 'option':
                    idxs = [i + len(so_data[name]['subtask_labels']) for i in idxs]
                idxs = [i for i in idxs if i not in seen]# HACK remove seen
                assert len(seen & set(idxs)) == 0
                seen.update(idxs)
                trajectory['subtask_indices'].extend(idxs)
                trajectory['subtask_indices'].extend(idxs)
                trajectory['start_flags'].extend([True] * len(idxs))
                trajectory['start_flags'].extend([False] * len(idxs))
            trajectory['subtask_indices'] = np.array(trajectory['subtask_indices'])
            trajectory['frame_numbers'] = np.arange(len(trajectory['subtask_indices']))
            trajectory['start_flags'] = np.array(trajectory['start_flags'])

            data['trajectories'].append(trajectory)
        data_by_task[name] = data
    return data_by_task

  def _augment_completion(self, data_by_task):
    # For a subtask completely missing in “subtask labels”, replace it with completion prediction.
    for task_name, data in data_by_task.items():
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']

      new_trajectories = []
      for trajectory in trajectories:
        new_trajectory = trajectory
        if 'completion_pred' in trajectory:
          subtask_indices = trajectory['subtask_indices'] # shape=[T]. np.array
          frame_numbers = trajectory['frame_numbers']     # shape=[T]. np.array
          start_flags = trajectory['start_flags']         # shape=[T]. np.array
          comp_pred = trajectory['completion_pred']       # shape=[T, num_subtask]. np.array

          items_to_add = []
          for subtask_idx in range(num_subtask):
            if subtask_idx not in subtask_indices:
              if comp_pred[:, subtask_idx].sum()>0:
                time_idx = np.argmax(comp_pred[:, subtask_idx])
                frame_num = frame_numbers[time_idx]
                items_to_add.append( (subtask_idx, frame_num) )

          for subtask_idx, frame_num in items_to_add:
            time_idx = np.argmax(frame_numbers == frame_num)
            # insert length-1 segment
            frame_numbers = np.concatenate((frame_numbers[:time_idx], [frame_num-1, frame_num], frame_numbers[time_idx:]))
            subtask_indices = np.concatenate((subtask_indices[:time_idx], [subtask_idx, subtask_idx], subtask_indices[time_idx:]))
            start_flags = np.concatenate((start_flags[:time_idx], [True, False], start_flags[time_idx:]))
          new_trajectory['subtask_indices'] = subtask_indices
          new_trajectory['frame_numbers'] = frame_numbers
          new_trajectory['start_flags'] = start_flags

        new_trajectories.append(new_trajectory)
      data_by_task[task_name]['trajectories'] = new_trajectories
    return data_by_task

  def _to_segment_format(self, trajectory):
    new_trajectory = trajectory
    subtask_indices = []
    start_flags = []
    frame_numbers = []
    fr_num = 0
    for subtask_idx in trajectory['subtask_indices']:
      # start
      subtask_indices.append(subtask_idx)
      start_flags.append(True)
      frame_numbers.append(fr_num)
      fr_num = fr_num + 2
      # end
      subtask_indices.append(subtask_idx)
      start_flags.append(False)
      frame_numbers.append(fr_num)
      fr_num = fr_num + 8
    new_trajectory['subtask_indices'] = np.array(subtask_indices)
    new_trajectory['start_flags'] = np.array(start_flags)
    new_trajectory['frame_numbers'] = np.array(frame_numbers)
    return new_trajectory

  def _remove_duplication(self, data):
    num_subtask = data['num_subtask']
    trajectories = data['trajectories']

    # only leave the first one
    new_trajectories = []
    for trajectory in trajectories:
      subtask_indices = trajectory['subtask_indices']
      num_steps = len(subtask_indices)

      # Set mask = 0 if the count >=2 (i.e., start->end->start again)
      subtask_count = np.zeros((num_subtask,), dtype=int)
      mask = np.zeros((num_steps,), dtype=bool)
      for ind, subtask_index in enumerate(subtask_indices):
        if subtask_count[subtask_index] < 2:
          subtask_count[subtask_index] = subtask_count[subtask_index] + 1
          mask[ind] = 1
      new_trajectory = dict()
      for key, val in trajectory.items():
        if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
          new_trajectory[key] = val # just copy
        else:
          if len(val) == len(mask):
            new_trajectory[key] = val[mask] # mask-out duplicated ones
          else:
            new_trajectory[key] = val # just copy

      new_trajectories.append(new_trajectory)

    data['trajectories'] = new_trajectories
    return data

  def _check_data(self):
    # remove empty trajectory
    for task_name, data in self.data_by_task.items():
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']
      data['trajectories'] = [trajectory for trajectory in trajectories if len(trajectory['subtask_indices'])>0] # remove empty trajectory

    # change to segment data if not
    for task_name, data in self.data_by_task.items():
      trajectories = data['trajectories']
      for traj_idx, trajectory in enumerate(trajectories):
        if 'start_flags' not in trajectory:
          trajectory = self._to_segment_format(trajectory)

    # remove duplication
    for task_name, data in self.data_by_task.items():
      data = self._remove_duplication(data)

    for task_name, data in self.data_by_task.items():
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']

      # data dimension consistency
      for traj_idx, trajectory in enumerate(trajectories):
        assert isinstance(trajectory, dict), "trajectory must be dict"
        assert len(trajectory['subtask_indices']) == len(trajectory['start_flags']), "size mismatch in trad id {}".format(traj_idx)
        assert len(trajectory['subtask_indices']) == len(trajectory['frame_numbers']), "size mismatch in trad id {}".format(traj_idx)
        assert trajectory['subtask_indices'].dtype == np.int_ or trajectory['subtask_indices'].dtype == np.int32, "data type error!"
        assert trajectory['start_flags'].dtype == np.bool_, "data type error! in trad id {}".format(traj_idx)
        assert trajectory['frame_numbers'].dtype == np.int_ or trajectory['frame_numbers'].dtype == np.int32, "data type error! in trad id {}".format(traj_idx)
        has_started = np.zeros(num_subtask, dtype=np.bool_)
        for start_flag, subtask_index in zip(trajectory['start_flags'], trajectory['subtask_indices']):
          if start_flag:
            if has_started[subtask_index]:
              print(trajectory['subtask_indices'])
              print(trajectory['start_flags'])
              print(trajectory['frame_numbers'])
              print(f'subtask {subtask_index} started twice')
            assert not has_started[subtask_index], "error in start flag in trad id {}".format(traj_idx)
            has_started[subtask_index] = True
          else:
            if not has_started[subtask_index]:
              print(task_name)
              print(trajectory['subtask_indices'])
              print(trajectory['start_flags'])
              print(trajectory['frame_numbers'])
              print(f'subtask {subtask_index} ended before start or ended twice')
            assert has_started[subtask_index], "error in start flag in trad id {}".format(traj_idx)
            has_started[subtask_index] = False

  def get_data_by_task(self, task_name):
    if task_name in self.task_names:
      return self.data_by_task[task_name]
    else:
      print(f'Error!! invalid task name: {task_name}')
      return None

  def to_order_data(self):
    order_data_by_task = dict()
    for task_name, data in self.data_by_task.items():
      if task_name == 'make_salmon_sandwich':
        continue
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']
      subtask_labels = data['subtask_labels']
      assert type(trajectories) == list, f"input trajectories should be a list type but is {type(trajectories)} type"

      # 0. get precedent data
      first_count = np.zeros((num_subtask), dtype=np.int_)
      bigram_count = np.zeros((num_subtask, num_subtask), dtype=np.int_)
      precedent_count = np.zeros((num_subtask, num_subtask), dtype=np.int_)
      precedent_weighted_count = np.zeros((num_subtask, num_subtask), dtype=np.float_)
      for trajectory in trajectories:
        subtask_indices = trajectory['subtask_indices']
        frame_numbers = trajectory['frame_numbers']
        start_flags = trajectory['start_flags']

        # fill-out first
        first_count[subtask_indices[0]] = first_count[subtask_indices[0]] + 1

        # fill-out precedent
        for t_next, curr_idx in enumerate(subtask_indices):
          for t_prev in range(t_next):
            if frame_numbers[t_prev] < frame_numbers[t_next] and not start_flags[t_prev] and start_flags[t_next]:
              prev_idx = subtask_indices[t_prev]
              if prev_idx == curr_idx:
                print('Error! ')
                import ipdb; ipdb.set_trace()
              precedent_count[prev_idx, curr_idx] = precedent_count[prev_idx, curr_idx] + 1
              if t_prev+1 == t_next:
                precedent_weighted_count[prev_idx, curr_idx] = precedent_weighted_count[prev_idx, curr_idx] + 2
              elif t_prev+2 == t_next:
                precedent_weighted_count[prev_idx, curr_idx] = precedent_weighted_count[prev_idx, curr_idx] + 1.5
              else:
                precedent_weighted_count[prev_idx, curr_idx] = precedent_weighted_count[prev_idx, curr_idx] + 1

        # fill-out bigram
        recent_done_subtask_idx = None
        for t, current_subtask_idx in enumerate(subtask_indices):
          if recent_done_subtask_idx is not None and start_flags[t]:
            prev_idx = recent_done_subtask_idx
            curr_idx = subtask_indices[t]
            bigram_count[prev_idx, curr_idx] = bigram_count[prev_idx, curr_idx] + 1
          if not start_flags[t]:
            recent_done_subtask_idx = subtask_indices[t]

      order_data = dict(
        num_subtask=num_subtask,
        subtask_labels=subtask_labels,
        bigram_count=bigram_count,
        precedent_count=precedent_count,
        precedent_weighted_count=precedent_weighted_count,
        first_count=first_count,
        num_trajectories=len(trajectories),
      )
      order_data_by_task[task_name] = order_data
    return order_data_by_task

  def to_eval_data(self):
    # fillout completion from trajectory
    eval_data_by_task = dict()
    for task_name, data in self.data_by_task.items():
      if task_name == 'make_salmon_sandwich':
        continue
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']

      eval_data_by_traj = []
      for traj_id, trajectory in enumerate(trajectories):
        subtask_indices = trajectory['subtask_indices']
        start_flags = trajectory['start_flags']

        if False:
          current_comp = np.ones((num_subtask), dtype=bool)
          current_comp[list(set(subtask_indices))] = False # TODO: improve by considering its precond
        else:
          current_comp = np.zeros((num_subtask), dtype=bool)

        comp_list = [current_comp]
        next_subtask_indices = []
        for subtask_idx, is_start in zip(subtask_indices, start_flags):
          if is_start:
            pass
          else:
            next_comp = current_comp.copy()
            assert not next_comp[subtask_idx], "Error in the code!!"
            next_comp[subtask_idx] = True
            comp_list.append(next_comp)
            current_comp = next_comp
            next_subtask_indices.append(subtask_idx)

        completion_mat = np.stack(comp_list[:-1])
        assert completion_mat.shape[0] == len(next_subtask_indices)
        eval_data_dict = dict(
          completion=completion_mat,
          next_subtask_indices=next_subtask_indices,
        )
        eval_data_by_traj.append(eval_data_dict)
      eval_data_by_task[task_name] = eval_data_by_traj
    return eval_data_by_task

  def to_comp_elig_data(self):
    # fillout completion from trajectory
    comp_elig_data_by_traj_by_task = dict()
    for task_name, data in self.data_by_task.items():
      if task_name == 'make_salmon_sandwich':
        continue
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']

      comp_elig_data_by_traj = []
      for traj_id, trajectory in enumerate(trajectories):
        subtask_indices = trajectory['subtask_indices']
        start_flags = trajectory['start_flags']

        total_steps = len(subtask_indices)+1
        comp_mat = np.zeros((total_steps, num_subtask), dtype=bool)
        elig_mat = np.zeros((total_steps, num_subtask), dtype=bool)
        for t in range(total_steps-1):
          subtask_idx = subtask_indices[t]
          is_start = start_flags[t]
          if is_start: # subtask #(subtask_idx) is eligible (might have been eligible earlier)
            elig_mat[t+1:, subtask_idx] = True
          else: # subtask #(subtask_idx) just became complete
            comp_mat[t+1:, subtask_idx] = True
        if np.any(np.all(comp_mat[:-1], axis=1)):
          import ipdb; ipdb.set_trace()
          print('Error! all the completion is True')
        comp_elig_data = dict(
          completion=comp_mat,
          eligibility=elig_mat,
        )
        comp_elig_data_by_traj.append(comp_elig_data)
      comp_elig_data_by_traj_by_task[task_name] = comp_elig_data_by_traj
    return comp_elig_data_by_traj_by_task

  def filter_trajectory(self, filter_threshold_fraction=0):
    if filter_threshold_fraction <= 0 or filter_threshold_fraction >= 1.0:
      return False
    num_traj_prev = 0
    num_traj_new = 0
    for task_name, data in self.data_by_task.items():
      if task_name == 'make_salmon_sandwich':
        continue
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']
      subtask_labels = data['subtask_labels']

      num_trajectory = len(trajectories)
      num_traj_prev = num_traj_prev + num_trajectory
      filter_threshold = int(filter_threshold_fraction * num_trajectory)
      if num_trajectory >= filter_threshold:
        score_by_traj = [trajectory[self.score_key] for trajectory in trajectories]
        score_threshold = np.sort(score_by_traj)[num_trajectory - filter_threshold]
        new_trajectories = [trajectory for trajectory in trajectories if trajectory[self.score_key]>= score_threshold]
      else:
        new_trajectories = trajectories
      num_traj_new = num_traj_new + len(new_trajectories)
      # dump
      self.data_by_task[task_name]['trajectories'] = new_trajectories
    print(f'Before filtering={num_traj_prev}')
    print(f'After filtering={num_traj_new}')

    return True


  def to_so_ilp_data(self):
    so_ilp_by_task = {}
    for task_name, data in self.so_data_by_task.items():
      N = data['num_subtask']
      M = data['num_options']
      completion = []
      options = []
      for trajectory in data['trajectories']:
        T = len(trajectory['subtask_and_option_indices'])
        assert T % 2 == 0
        comp = np.zeros((T // 2, N), dtype=np.bool_)
        ops = np.zeros((T // 2, M), dtype=np.bool_)
        expect_option = True
        for _t, (turn_type, indices) in enumerate(trajectory['subtask_and_option_indices']):
          t = _t // 2
          assert (turn_type == 'option') if expect_option else (turn_type == 'subtask')

          if expect_option:
            for idx in indices:
              ops[t, idx] = True
          else:
            if t > 0:
              comp[t] = comp[t - 1]# completion remains across timesteps
            for idx in indices:
              comp[t, idx] = True

          expect_option = not expect_option
        assert expect_option# ends on a subtask
        completion.append(comp)
        options.append(ops)
      so_ilp = dict(
        num_subtask=N,
        num_option=M,
        subtask_labels=data['subtask_labels'],
        option_labels=data['option_labels'],
        completion=completion,
        options=options
      )
      so_ilp_by_task[task_name] = so_ilp
    return so_ilp_by_task

  def to_ilp_data(self):
    ilp_data_by_task = dict()
    for task_name, data in self.data_by_task.items():
      if task_name == 'make_salmon_sandwich':
        continue
      num_subtask = data['num_subtask']
      trajectories = data['trajectories']
      subtask_labels = data['subtask_labels']

      # trajectories: [trajectory, ...]
      ### trajectory: dict(subtask_indices, subtask) -- if segmented
      ### trajectory == subtask_indices              -- otherwise
      assert type(trajectories) == list, f"input trajectories should be a list type but is {type(trajectories)} type"

      # 0. get precedent data
      num_trajectory = len(trajectories)
      soft_elig_by_traj_by_subtask = np.zeros((num_subtask, num_trajectory, num_subtask))
      precedent_set_list = [set() for _ in range(num_subtask)]

      if self.score_key:
        score_by_traj = [trajectory[self.score_key] for trajectory in trajectories]
        min_score = min(score_by_traj)
        max_score = max(score_by_traj)
        weight_by_traj = [ (score - min_score) / (max_score - min_score) for score in score_by_traj] # range(0, 1)
      else:
        weight_by_traj = [1] * len(trajectories)

      for traj_id, trajectory in enumerate(trajectories):
        subtask_indices = trajectory['subtask_indices']
        frame_numbers = trajectory['frame_numbers']
        start_flags = trajectory['start_flags']

        for t, current_subtask_idx in enumerate(subtask_indices):
          if start_flags[t]: # eligibility for current subtask => start
            precedent_dones = subtask_indices[:t][~start_flags[:t]]
            precedent_dones_set = set(precedent_dones.flatten())
            precedent_set_list[current_subtask_idx].update(precedent_dones_set)
            step_curr = len(precedent_dones)
            for step_prev, prev_subtask_idx in enumerate(precedent_dones):
              #soft_elig = max(1.15 - 0.15 * (step_curr - step_prev), 0.3)
              soft_elig = max(pow(0.7,(step_curr - step_prev-1)), 0.1)
              soft_elig_by_traj_by_subtask[current_subtask_idx][traj_id][prev_subtask_idx] = soft_elig

      # 0. expand and truncate trajectory by subtask (handle segments)
      traj_by_subtask, weight_history_by_subtask = self._expand_trajectory(trajectories, num_subtask, weight_by_traj)

      # 1. fillout positive eligibility data
      pos_comp_list, no_precond_by_subtask, pos_score_by_subtask, weight_by_subtask = self._fill_positive_eligibility(traj_by_subtask, num_subtask, weight_history_by_subtask)
      pos_elig_by_subtask = [np.array([True]*len(pos_comp))  for pos_comp in pos_comp_list]

      # 2. fillout negative eligibility data
      neg_comp_list = self._fill_negative_eligibility(traj_by_subtask, pos_comp_list)
      neg_elig_by_subtask = [np.array([False]*len(neg_comp))  for neg_comp in neg_comp_list]

      # 3. merge to construct ilp data
      neg_comp_by_subtask, pos_comp_by_subtask = [], []
      for subtask_idx in range(num_subtask):
        if len(pos_comp_list[subtask_idx]) == 0: # no precondition
          pos_comp_by_subtask.append(np.array([True]))
        else:
          pos_comp_by_subtask.append(np.stack(pos_comp_list[subtask_idx]))
        if len(neg_comp_list[subtask_idx]) == 0:
          neg_comp_by_subtask.append(None)
        else:
          neg_comp_by_subtask.append(np.stack(neg_comp_list[subtask_idx]))

      # Store data
      for pos_comp, weight in zip(pos_comp_by_subtask, weight_by_subtask):
        if pos_comp.shape[-1] > 1:
          assert len(pos_comp) == len(weight)
      ilp_data = dict(
        num_subtask=num_subtask,
        subtask_labels=subtask_labels,
        completion=pos_comp_by_subtask,
        eligibility=pos_elig_by_subtask,
        weight=weight_by_subtask,
        neg_completion=neg_comp_by_subtask,
        neg_eligibility=neg_elig_by_subtask,
        traj_by_subtask=traj_by_subtask,
        precedent_set_list=precedent_set_list,
        soft_elig_by_traj_by_subtask=soft_elig_by_traj_by_subtask,
        weight_by_traj=weight_by_traj,
      )
      ilp_data_by_task[task_name] = ilp_data
    return ilp_data_by_task

  def _expand_trajectory(self, trajectories, num_subtask, weight_by_traj):
    traj_by_subtask = [[] for _ in range(num_subtask)]
    weight_history_by_subtask = [[] for _ in range(num_subtask)]
    if isinstance(trajectories[0], dict):
      for traj_id, trajectory in enumerate(trajectories):
        weight = weight_by_traj[traj_id]
        subtask_indices = trajectory['subtask_indices']
        start_flags = trajectory['start_flags']
        #
        comp_history = []
        for subtask_idx, is_start in zip(subtask_indices, start_flags):
          if is_start: # subtask #(subtask_idx) is eligible (might have been eligible earlier)
            traj_by_subtask[subtask_idx].append(np.array(comp_history)) # make a copy and convert to arr
            weight_history_by_subtask[subtask_idx].append(weight)
          else: # subtask #(subtask_idx) just became complete
            comp_history.append(subtask_idx)
    else:
      for traj_id, trajectory in enumerate(trajectories):
        weight = weight_by_traj[traj_id]
        subtask_indices = trajectory
        #
        comp_history = []
        for subtask_idx in subtask_indices:
          traj_by_subtask[subtask_idx].append(np.array(comp_history))
          weight_history_by_subtask[subtask_idx].append(weight)
          comp_history.append(subtask_idx)

    return traj_by_subtask, weight_history_by_subtask

  def _fill_positive_eligibility(self, traj_by_subtask, num_subtask, weight_history_by_subtask):
    comp_by_subtask, weight_by_subtask = [], []
    pos_score_by_subtask = []
    no_precond_by_subtask = []
    for subtask_idx, comp_histories in enumerate(traj_by_subtask):
      weight_history = weight_history_by_subtask[subtask_idx]
      comp_list, comp_code_list, weight_list = [], [], []
      score_list = []
      no_precond_flag = False
      for idx, comp_history in enumerate(comp_histories):
        weight = weight_history[idx]
        assert comp_history.ndim == 1
        comp_vec = logic_utils.to_multi_hot(comp_history, num_subtask)
        assert comp_vec.ndim == 1
        comp_code = logic_utils.batch_bin_encode(comp_vec)
        assert isinstance(comp_code, int)
        if comp_code not in comp_code_list:
          comp_list.append(comp_vec)
          weight_list.append(weight)
          comp_code_list.append(comp_code)
        if np.all(~comp_vec): # All false --> always eligible
          no_precond_flag = True
          break
      comp_by_subtask.append(comp_list)
      weight_by_subtask.append(weight_list)
      pos_score_by_subtask.append(score_list)
      no_precond_by_subtask.append(no_precond_flag)
    return comp_by_subtask, no_precond_by_subtask, pos_score_by_subtask, weight_by_subtask

  def _fill_negative_eligibility(self, traj_by_subtask, pos_comp_by_subtask):
    neg_comp_by_subtask = []
    for subtask_idx, comp_histories in enumerate(traj_by_subtask):
      pos_comp_list = pos_comp_by_subtask[subtask_idx]
      neg_comp_list = []
      comp_code_list = []
      for comp_history, pos_comp in zip(comp_histories, pos_comp_list):
        new_comp = pos_comp.copy()
        # incompatible > ... > compatible
        for completed_subtask_idx in np.flip(comp_history): # backtrack the completed subtasks
          new_comp[completed_subtask_idx] = False
          comp_code = logic_utils.batch_bin_encode(new_comp)
          if self._is_compatible(new_comp, pos_comp_list) and comp_code not in comp_code_list:
            neg_comp_list.append(new_comp)
            comp_code_list.append(comp_code)
            break
      neg_comp_by_subtask.append(neg_comp_list)
    return neg_comp_by_subtask

  def _is_compatible(self, neg_comp_to_test, pos_comp_list):
    num_pos = len(pos_comp_list)
    # neg     pos  > and           #pos  #and
    # 101     110  > 100: valid.   2   > 1
    # 111     110  > 110: invalid. 2   <=2
    # 110     110  > 110: invalid. 2   <=2
    # 100     110  > 100: valid.   2   > 1
    neg_comp_tiled = np.tile(neg_comp_to_test, [num_pos, 1]) # [num_pos x num_subtask]
    pos_comp_array = np.stack(pos_comp_list) # [num_pos x num_subtask]
    and_result = np.logical_and(neg_comp_tiled, pos_comp_array) # [num_pos x num_subtask]
    count_ones_in_and_result = and_result.sum(axis=-1) # [num_pos]
    count_ones_in_pos = pos_comp_array.sum(axis=-1) # [num_pos]
    is_invalid_list = (count_ones_in_pos <= count_ones_in_and_result) # [num_pos]

    return not np.any(is_invalid_list)
