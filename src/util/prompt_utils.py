import re

def subtask_options_to_prompt(subtasks_and_options, prompt_style=None, end=None):
    agent_set = ['USER', 'SYSTEM', 'DATABASE']
    if end is None:
        end = len(subtasks_and_options)
    assert end <= len(subtasks_and_options), f"argument 'end' is wrong. 'end'={end}"
    prompt = ''
    for i in range(0,end):
        turn_type = subtasks_and_options[i][0] # 'subtask' or 'option'
        assert turn_type in ['subtask', 'option'], "data has wrong format"
        label_list = subtasks_and_options[i][1]
        agent_label = [agent for agent in agent_set if agent in label_list[0]]
        if len(agent_label) > 0:
            agent_label = agent_label[0]
        else:
            agent_label = None
        assert agent_label in ['USER', 'SYSTEM', 'DATABASE', None]
        
        if 'symbolic' in prompt_style:
            processed_labels = [process_label(label, style='symbolic') for label in label_list]
            if turn_type == 'option':
                prompt += f"Actions: "
                prompt += ' | '.join(processed_labels)
            else: # subtask
                prompt += f" ; Statuses: "  # "User"/"System"
                prompt += ' | '.join(processed_labels)
                prompt += '\n'
        elif 'concise' in prompt_style:
            processed_labels = [process_label(label, style='concise') for label in label_list]
            if turn_type == 'option':
                prompt += f"{agent_label} act: "
                prompt += ', '.join(processed_labels)
                prompt += '\n'
            else: # subtask
                prompt += f"Status: "  # "User"/"System"
                if len(processed_labels) == 0:
                    prompt += f"No change"
                else:
                    prompt += ', '.join(processed_labels)
                prompt += '\n'
        else:
            raise NotImplementedError
    return prompt

def find_earliest(prompt, terminal_tokens):
    earliest_index = -1
    for token in terminal_tokens:
        if token in prompt:
            token_index = prompt.find(token)
            if earliest_index < 0:
                earliest_index = token_index
            elif earliest_index > token_index:
                earliest_index = token_index
    return earliest_index
            
    
def prompt_to_subtask_options(prompt, prompt_style):
    # USER_Inform number_of_rooms, Inform_Intent ReserveHotel
    # Status: Informed number of rooms, Intent Informed ReserveHotel
    # <->
    # USER Act: Inform number_of_rooms, Inform_Intent ReserveHotel
    # Status: Informed number of rooms, Intent Informed ReserveHotel

    prompt = prompt.lower()
    if 'concise' in prompt_style:
        option_token = "act:"
        subtask_token = "status:"
    elif 'symbolic' in prompt_style:
        option_token = "actions:"
        subtask_token = "statuses:"
    else:
        raise NotImplementedError

    # Remove all the sub-turn separator (i.e., separating between option and subtask)
    tokens_to_replace = [' ; ', ';', '\n', '   ', '  ']
    for token_to_replace in tokens_to_replace:
        prompt = prompt.replace(token_to_replace, ' ')
    
    # Split based on 'option' and 'subtask
    if option_token in prompt: # option exists!
        option_start_index = prompt.find(option_token)
        option_prompt = prompt[option_start_index:]
        option_prompt = re.split(f"{option_token}|{subtask_token}", option_prompt)[1]
        #
    else:
        # if none of option/subtask token found, then entire prompt is considered as actions
        option_prompt = re.split(f"{option_token}|{subtask_token}", prompt)[0]
    
    if subtask_token in prompt:
        subtask_start_index = prompt.find(subtask_token)
        subtask_prompt = prompt[subtask_start_index:]
        subtask_prompt = re.split(f"{option_token}|{subtask_token}", subtask_prompt)[1]
    else:
        subtask_prompt = ""
    
    # Parse option
    option_prompt = option_prompt.replace(option_token, "")
    prompt_chunks = re.split('\||,', option_prompt)
    option_list = []
    for prompt_chunk in prompt_chunks:
        label = prompt_chunk.strip() # remove leading/trailing white space
        if len(label) < 2:
            continue
        if 'concise' in prompt_style:
            label = "SYSTEM " + label
        elif 'symbolic' in prompt_style:
            label = label
        option_list.append(label)
    
    # Parse subtaskes
    subtask_prompt = subtask_prompt.replace(subtask_token, "")
    prompt_chunks = re.split('\||,', subtask_prompt)
    subtask_list = []
    for prompt_chunk in prompt_chunks:
        label = prompt_chunk.strip() # remove leading/trailing white space
        if len(label) < 2:
            continue
        if 'concise' in prompt_style:
            if "query success" in label:
                query = label.replace("query success", "").strip()
                label = f"STATUS database_n>0 {query}"
            elif "query fail" in label:
                query = label.replace("query fail", "").strip()
                label = f"STATUS database_n=0 {query}"
            elif "end" == label:
                label = "STATUS end of dialog"
            else:
                label = "STATUS SYSTEM " + label
        elif 'symbolic' in prompt_style:
            label = "STATUS " + label
        subtask_list.append(label)
    return subtask_list, option_list

def process_label(label, style):
    tokens_to_remove = ['[\'', '\']', 'STATUS_','STATUS ','status ']
    if 'concise' in style:
        tokens_to_remove += ['USER_', 'SYSTEM_','USER ','SYSTEM ']
    for token in tokens_to_remove:
        label = label.replace(token, '')
    
    if 'concise' in style:
        if "DATABASE_N>0" in label:
            query = label.replace("DATABASE_N>0", "").strip()
            label = f"{query} query success"
        elif "DATABASE_N=0" in label:
            query = label.replace("DATABASE_N=0", "").strip()
            label = f"{query} query fail"
        elif "END_OF_DIALOG" in label:
            label = "End"
    # lowercase except the verb (first word)
    index = label.find(' ') # between verb and objects
    if index >= 0:
        label = label[:index] + label[index:].lower()
    # remove '_'
    label = label.replace('_',' ')
    return label

def get_natural_task_name(task_name):
    if 's_' in task_name:
        return task_name[:task_name.index("s_")]
    elif '_' in task_name:
        return task_name[:task_name.index("_")]
    return task_name
