import torch
import copy, random
import os,sys
sys.path.append(os.getcwd())
from transformers import BertModel, BertTokenizer
from util.prompt_utils import subtask_options_to_prompt
import numpy as np
modelbert = None

def fetch_embed(name,num,trajdict):
    return trajdict[name][0][num]

def find_closest_in_train(embed,trajdict,num_samples):
    maxses = []
    for name in trajdict:
        if trajdict[name][1] != 'train':
            continue
        embeds = np.array(trajdict[name][0])
        maxs = np.max(embeds @ embed)
        maxses.append((maxs,name))
    maxses.sort()
    ret = []
    for i in range(num_samples):
        ret.append((maxses[-1-i][1]))
    return ret

# same as demo_trajectory_sampler stuff
def processdemo(org_traj):
    demo_traj = copy.deepcopy(org_traj)
        
    assert len(demo_traj['subtasks_and_options']) == len(demo_traj['subtask_and_option_indices'])
    subtasks_and_options = demo_traj['subtasks_and_options']
        
    # Sample random system option turn
    system_option_turn_indices = []
    for turn_index, turn_tuple in enumerate(subtasks_and_options):
        utterance_concat = "".join(turn_tuple[1])
        if turn_tuple[0]=='option' and 'system' in utterance_concat.lower():
            system_option_turn_indices.append(turn_index)
    rand_system_option_turn_index = random.choice(system_option_turn_indices)
    
    # Construct demo trajectory
    demo_traj['turn_index'] = rand_system_option_turn_index
    ### truncate until the randomly sampled turn index (i.e., history)
    demo_traj['entire_subtasks_and_options'] = org_traj['subtasks_and_options']
    demo_traj['entire_subtask_and_option_indices'] = org_traj['subtask_and_option_indices']
    #
    demo_traj['subtasks_and_options'] = demo_traj['subtasks_and_options'][:rand_system_option_turn_index]
    demo_traj['subtask_and_option_indices'] = demo_traj['subtask_and_option_indices'][:rand_system_option_turn_index]
    ### Add gt label
    demo_traj['gt_system_option'] = org_traj['subtasks_and_options'][rand_system_option_turn_index]
    demo_traj['gt_system_option_index'] = org_traj['subtask_and_option_indices'][rand_system_option_turn_index]

    return demo_traj

#GPT-4 codes and comments lol
def get_bert_embeddings(sentence):
    # Check if a GPU is available and if not, we'll use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global modelbert
    if modelbert is None:
        modelbert = BertModel.from_pretrained('bert-base-uncased')
        modelbert = modelbert.to(device)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_ids = torch.tensor([input_ids]).to(device)  # Create a torch tensor and send it to the device

    # Make sure the model is in evaluation mode (it's also the default after initialization)
    modelbert.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = input_ids.to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        outputs = modelbert(tokens_tensor)
        # Transformers models always output tuples. See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the BERT model
        encoded_layers = outputs.last_hidden_state

    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    assert tuple(encoded_layers.shape) == (1, len(input_ids[0]), modelbert.config.hidden_size)

    # We take the mean of the sequence dimension to get an sentence embedding
    sentence_embedding = torch.mean(encoded_layers, dim=1).squeeze()

    return sentence_embedding.cpu().numpy()  # return the embeddings back to cpu in numpy format

import numpy as np
from GPT.OpenAIAPIembedding import getembedding

def compute_traj_embeddings(traj):
    toembeds = []
    for i in range(2,len(traj['subtasks_and_options'])):
        if traj['subtasks_and_options'][i][0] == 'option' and traj['subtasks_and_options'][i][1][0][0:6] == 'SYSTEM':
            prompt = subtask_options_to_prompt(traj['subtasks_and_options'],'concise',i)
            toembeds.append(prompt)
    #print(toembeds)
    return getembedding(toembeds)

import json
from tqdm import tqdm
def convert_embeddings(fn):
    outname = fn.replace('_trajectories.json','_demo_embeds.pt')
    trajs = json.load(open(fn))
    for tn in trajs:
        pass
    trajs = trajs[tn]['trajectories']
    ret = {}
    for traj in tqdm(trajs):
        embeds = compute_traj_embeddings(traj)
        ret[traj['name']] = (embeds,traj['split'])
    torch.save(ret,outname)

if __name__ == '__main__':
    rootdir = '../datasets/SGD/trajectories/'
    count = 0
    for fn in os.listdir(rootdir):
        if 'trajectories.json' in fn:
            count += 1
            if count >= 21 or count < 11:
                continue
            print('Converting '+fn)
            convert_embeddings(rootdir+fn)

    

