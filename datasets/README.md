## JSON explanations

Dictionary Structure of Relabeled Data: (example: outputs/sample_data_2.json)

```
['<schema and task name>']: {
                             'option_labels': ['SYSTEM_Request destination', ...]          (possible option labels)
                             'subtask_labels': ['STATUS_SYSTEM_Request destination', ...]  (possible subtask labels)
                             'num_options': ...,                                           (total number of options)
                             'num_subtasks': ...,                                          (total number of subtasks)
                             'trajectories': [                                             (all trajectories in dataset)
                                             {
                                             'name': '41_00013'                            (name/id of the trajectory)
                                             'split':'train'                               (split of the trajectory, 'train' or 'val')
                                             'subtask_and_option_indices': [
                                                       ['subtask',[0,1,2]],
                                                       ['option',[0,1]],...]                (subtask and option indices of each time step)
                                             'subtasks_and_options': [
                                                 [
                                                     "option",                              (subtask or option)
                                                     ['USER_Inform number of days', ...],   (list of subtasks/options in text)
                                                     'I want to book a hotel for 3 days.',  (The utterance)
                                                     {'number of days': 3, ...}             (Dictionary of updated slots in this step. Only options have this)
                                                 ],
                                                 ['subtask', ...],...]
                                             },...
                                             ]
                            }
```

NPY dictionary object of Graph representation: (example: outputs/gt_graph (!).npy)

```

{
       'option_labels': ['SYSTEM_Request destination', ...]             (possible option labels)
       'subtask_labels': ['STATUS_SYSTEM_Request destination', ...]     (possible subtask labels)                    
       (Note that in each of the matrices below, the matrices are transposed, i.e. m[i][j]==1 indicates there is an edge from node j to node i)
       'CAN':     {                                                     (Preconditions)
                       'AndMat':  Num_And_Nodes x Num_Subtasks matrix   (And relationships in preconditions, links from subtasks to and-nodes, 1 means positive (can) and -1 means negative (cannot))
                       'OrMat': Num_Options x Num_And_Nodes matrix      (Or relationships in preconditions, links from and-nodes to options)               
                  }
       'SHOULD':  {                                                     (Should relationships)
                       'soft_should': Num_options x Num_subtasks matrix (Soft Should relationships, links subtasks to options: indicating the next speaker needs to choose one of the options when a certain subtask status is active)
                       'hard_should': Num_options x Num_subtasks matrix (Hard Should relationships, links subtasks to options: indicating that the next speaker should perform certain options when the subtask status is active)
                  }
       'Effect':  {
                       'soft_effect': Num_options x Num_options         (Soft effect relationships, performing the option could lead to one of several subtask statuses)
                       'hard_effect': Num_options x Num_options         (Hard effect relationships, performing the option leads to subtasks as effects)
                  }
}

```

Dictionary structure of mapping configs: (example: src/relabling_mapping/hotel_mapping_v2.json)

```
{
'<Action Annotation Name>': {
                             'generalmap': '<SPEAKER>_Inform <SLOT>'                     (maps the action to an option/subtask of following the string format.)
                             'usermap': 'USER_No_More_Need'                              (optional: this will only be used when action is performed by user. Must not specify generalmap when using this. 'systemmap' should be specified together with this)
                             'systemmap': 'SYSTEM_Goodbye'                               (optional: this will only be used when action is performed by system. Must not specify generalmap when using this. 'usermap' should be specified together with this)
                             'intentcondition': 'NONE'                                   (optional: specify this to only relabel when the active intent matches the condition)
                             'onlyonce': True                                            (optional: specify this if you only want this action to be relabeled once per time step regardless of how many such actions were annotated)
                            }
...
}
```
