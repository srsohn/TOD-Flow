import json
import random
import copy
import sys,os
sys.path.append(os.getcwd())

def processstr(st):
    st = st.replace('[\'','')
    st = st.replace('\']','')
    st = st.replace('STATUS_','')
    st = st.replace('_',' ')
    return st


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
    return res


def getalldemotrajs(trajs):
    elitrajs = []
    for traj in trajs:
        if traj['split']=='train':
            elitrajs.append(traj)
    return elitrajs

def getdemo(demotrajs,num_demos):
    trajs = random.sample(demotrajs,num_demos)
    ret = ''
    for traj in trajs:
        ret += 'Demonstration:\n'+trajform(traj['subtasks_and_options'])+'\n'
    return ret

def getactionprompt(historystr,candstr,nsample):
    prompt = getdemo(alldemotrajs,nsample)
    prompt += '\n'+'Now consider the following partial dialog:\n\n'
    prompt += historystr
    prompt += '\nBased on the demonstrations above, predict the next SYSTEM response:\nSYSTEM:'
    #prompt += candstr
    #prompt += '\n SYSTEM Actions: SYSTEM'
    #print(prompt)
    return prompt

from GPT.OpenAIAPIchat import getcompletion
def getaction(historystr,candstr):
    for i in range(8):
        try:
            out = getcompletion(getactionprompt(historystr,candstr,14-i),'turbo',0)
            break
        except Exception as e:
            print('Unfortunate '+str(i))
            print(e)
            pass
    return out

from tqdm import tqdm
namewoz = sys.argv[1]
a = json.load(open('../datasets/MultiWOZ/trajectories/'+namewoz+'_trajectories.json'))[namewoz.lower()]['trajectories']
alldemotrajs = getalldemotrajs(a)
ret = {}
for traj in tqdm(a):
    if traj['split'] == 'test':

        ret[traj['name']] = []
        for i in range(len(traj['subtasks_and_options'])//4):
            historystr = trajform(traj['subtasks_and_options'],i*4+2)
            outs = getaction(historystr,'')
            ret[traj['name']].append(outs)
json.dump(ret,open('chat'+namewoz.lower()+'Test.json','w+'),indent=2)









