import json
import random
import copy
import sys,os
sys.path.append(os.getcwd())

# standardize string for GPT
def processstr(st):
    st = st.replace('[\'','')
    st = st.replace('\']','')
    st = st.replace('STATUS_','')
    st = st.replace('_',' ')
    return st

# convert a trajectory into a GPT demonstration
def trajform(trajectory, end = None):
    res = ''
    if end is None:
        end = len(trajectory)
    for i in range(0,end):
        c = trajectory[i]
        speaker = 'USER'
        if i % 4 >= 2:
            speaker = 'SYSTEM'
        if c[0] == 'option':
            res += speaker+' Utterance: '+c[3]+'\n'
            if speaker == 'SYSTEM':
                reses = []
                for op in c[1]:
                    reses.append(processstr(op))
                res += ('SYSTEM Actions: ' + ' | '.join(reses)+'\n')
    return res

# Get all demonstrations
def getalldemotrajs(trajs):
    elitrajs = []
    for traj in trajs:
        if traj['split']=='train':
            elitrajs.append(traj)
    return elitrajs

# Get one demonstration
def getdemo(demotrajs,num_demos):
    trajs = random.sample(demotrajs,num_demos)
    ret = ''
    for traj in trajs:
        ret += 'Demonstration:\n'+trajform(traj['subtasks_and_options'])+'\n'
    return ret

# Get the entire prompt for GPT
def getactionprompt(historystr,candstr,nsample):
    prompt = getdemo(alldemotrajs,nsample)
    prompt += '\n Now, we are going to ask you to predict the actions from several candidate responses. Your answer format must fit the following example:'
    prompt += '\n\n(1) [value_name] is a [value_food] restaurant .\n(2) there are [value_choice] restaurant-s that meet your criteria . 1 serves [value_food] food and the other serves [value_food] food . do you have a preference ?\n(3) there are [value_choice] restaurant-s that meet that criteria . 1 serves [value_food] food and the other serves [value_food] food . which would you prefer ?'
    prompt += '\n Your output should be:\n(1) SYSTEM restaurant-inform name | SYSTEM restaurant-inform food\n(2) SYSTEM restaurant-ask-for-selection food | SYSTEM restaurant-inform num-choices | SYSTEM restaurant-inform food \n(3) SYSTEM restaurant-ask-for-selection food | SYSTEM restaurant-inform num-choices | SYSTEM restaurant-inform food'
    prompt += '\n'+'Now consider the following partial dialog:\n\n'
    prompt += historystr
    prompt += '\nBased on the demonstrations above, predict the SYSTEM Actions for each of the following candidate SYSTEM response:\n'
    prompt += candstr
    prompt += '\n\nYour Answer:\n(1)'
    return prompt

import time
from GPT.OpenAIAPIchat import getcompletion

# Get Action annotation from GPT using the prompt from above
def getaction(historystr,candstr):
    for i in range(5):
        try:
            out = getcompletion(getactionprompt(historystr,candstr,6-i),'turbo',0,500)
            out = '(1) '+out
            break
        except Exception as e:
            print('Unfortunate '+str(i))
            print(e)
            if ('Rate limit' in str(e)) or ('Bad gateway' in str(e)):
                time.sleep(60)
            pass
    return out

# post-processing of the outputs
def prep(s):
    s = s.lower()
    s = s.replace('[attraction_','[value_')
    s = s.replace('[restaurant_','[value_')
    s = s.replace('[taxi_','[value_')
    s = s.replace('[hotel_','[value_')
    s = s.replace('[train_','[value_')
    s = s.replace('[value_trainid','[value_id')
    return s


from tqdm import tqdm
namedict = {'galaxy':'../datasets/MultiWOZ/e2e/galaxy_7full_pred.json','hdno':'../datasets/MultiWOZ/e2e/hdno_7_pred.json','hdsa':'../datasets/MultiWOZ/e2e/hdsa_7new_pred.json'}
namewoz = sys.argv[1]
a = json.load(open('../datasets/MultiWOZ/trajectories/'+namewoz+'_trajectories.json'))[namewoz.lower()]['trajectories']
alldemotrajs = getalldemotrajs(a)
b1 = json.load(open(namedict[sys.argv[2].lower()]))

ret = {}
for traj in tqdm(a):
    if traj['split'] == 'test':
        preds = b1[traj['name'].lower()[:-5]]

        ret[traj['name']] = []
        assert len(traj['subtasks_and_options']) == 4 * len(preds)
        for i in range(len(preds)):
            historystr = trajform(traj['subtasks_and_options'],i*4+2)
            cands = preds[i]
            candstr = ''
            for j in range(len(cands)):
                candstr += '('+str(j+1)+') '+prep(cands[j])+'\n'
            outs = getaction(historystr,candstr)
            ret[traj['name']].append(outs)
    json.dump(ret,open(namewoz.lower()+'tmpTest.json','w+'),indent=2)









