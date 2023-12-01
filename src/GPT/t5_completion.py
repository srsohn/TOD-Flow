from transformers import pipeline, set_seed
import torch
import time
import numpy as np
from joblib import Memory
from GPT.GPTutils import NUM_SAMPLES
from util.prompt_utils import prompt_to_subtask_options
from util.eval_utils import edit_distance
memory = Memory("cachedir", verbose=0)

NUM_MAX_ACTIONS_PER_TURN = 5
MAX_ACTIONS = 30
MAX_TOKEN_OF_SINGLE_ACTION = 6
MAX_RETRIES = 30
_generator = None# cache load the LM when needed

#@memory.cache
def getcompletion(prompt, model='flan-t5-xxl', temperature=0, max_length=MAX_ACTIONS, num_return_sequences=NUM_SAMPLES):
    global _generator
    if _generator is None:
        print(f'Loading LLM model {model}')
        t0 = time.time()
        _generator = pipeline(
            'text2text-generation',
            model=f'google/{model}',
            device="cuda:0",
            torch_dtype=torch.bfloat16,
            #model_kwargs={'cache_dir': HF_CACHE}
        )
        print(f'Done. Took {time.time()-t0} sec')
    
    if temperature == 0:
        completions = _generator(prompt, max_length=max_length, temperature=0)
        return completions[0]['generated_text']
    else: # multi-sampling
        batch_size = max(60 // max_length, 1)
        completions = []
        for _ in range(num_return_sequences):
            output = _generator(prompt, max_length=max_length, temperature=temperature, do_sample=True, num_return_sequences=batch_size)
            completions.extend(output)
            if len(completions) >= num_return_sequences:
                break
        completions = completions[:num_return_sequences]
        response_list = [completion['generated_text'] for completion in completions]
        return response_list

def sample_action_by_action(prompt, model, temperature, elig_actions, prompt_style):
    current_prompt = prompt
    separator = ', ' if 'concise' in prompt_style else ' | '
    response = ""
    for i in range(NUM_MAX_ACTIONS_PER_TURN):
        # 1. sample MAX_RETRIES times with length=MAX_TOKEN_OF_SINGLE_ACTION
        #print('prompt=',current_prompt.split('\n')[-1])
        #print(f'{i}-th prompt len', len(current_prompt))
        response_list = getcompletion(current_prompt, model, temperature, max_length=MAX_TOKEN_OF_SINGLE_ACTION, num_return_sequences=MAX_RETRIES)
        assert len(response_list) == MAX_RETRIES, "error in getcompletion"
        
        # 2. extract first action for each. filter-out ineligible actions
        actions = extract_first_action(response_list, prompt_style)
        filtered_actions = filter_action(actions, elig_actions)
        
        # 3. Do majority voting. If filtered actions are all empty, end the generation. If not, add it to prompt, repeat.
        if all([act is None for act in filtered_actions]):
            break
        major_action = find_majority(filtered_actions)
        if 'concise' in prompt_style:
            major_action = major_action.replace('SYSTEM ', '')
        
        append_text = separator + major_action if i > 0 else major_action
        #print(f'{i}-th action: {major_action}')
        current_prompt += append_text
        response += append_text
    return response

def extract_first_action(response_list, prompt_style):
    actions = []
    for response in response_list:
        _, option_list = prompt_to_subtask_options(response, prompt_style)
        if len(option_list) > 0:
            first_action = option_list[0]
            actions.append(first_action)
        else:
            actions.append(None)
    return actions

def filter_action(actions, elig_actions, approx_thresh=0.05):
    filtered_actions = []
    for action in actions:
        if action is None:
            filtered_actions.append(None)
            continue
        dists = [edit_distance(elig_action, action) for elig_action in elig_actions]
        closest_i = np.argmin(dists)
        min_dist = dists[closest_i]
        if (min_dist / len(action)) < approx_thresh:
            filtered_actions.append(action)
        else:
            filtered_actions.append(None)
    return filtered_actions

def find_majority(actions):
    from collections import Counter
    import random
    actions_except_none = [act for act in actions if act is not None]
    count_result = Counter(actions_except_none)
    sorted_action_count = count_result.most_common()
    max_count = sorted_action_count[0][1]
    most_common_actions = [val_count[0] for val_count in sorted_action_count if val_count[1] == max_count]
    return random.choice(most_common_actions)