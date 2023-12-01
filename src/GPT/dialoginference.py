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
        if c[0] == 'option':
            if 'SYSTEM_Query' in c[1][0]:
                continue
            speaker = 'SYSTEM'
            if c[1][0][0:4] == 'USER':
                speaker='USER'
            res += speaker+' Utterance: '+c[2]+'\n'
            reses = []
            for op in c[1]:
                reses.append(processstr(op))
            res += ('Actions: ' + ' | '.join(reses))
        else:
            if len(c[1])>0 and c[1][0][7:15] == 'DATABASE':
                continue
            reses = []
            for op in c[1]:
                reses.append(processstr(op))
            res += (' ; Statuses: ' + ' | '.join(reses) + '\n')
    return res

def trajform2(trajectory, end = None):
    res = ''
    if end is None:
        end = len(trajectory)
    for i in range(0,end):
        c = trajectory[i]
        if c[0] == 'option':
            speaker = 'SYSTEM'
            if c[1][0][0:4] == 'USER':
                speaker='USER'
            if 'SYSTEM_Query' in c[1][0]:
                continue
            res += speaker+' Utterance: '+c[2]+'\n'
    return res

def randomtraj(trajpath,taskname, num, exclude = None,settrajs = None):
    s = json.load(open(trajpath))
    if taskname is None:
        taskname = list(s.keys())[0]
    trajs = s[taskname]['trajectories']
    if exclude is not None:
        for d in trajs:
            if d['name'] == exclude:
                trajs.remove(d)
                break
    setted = []
    rtrajs = copy.deepcopy(trajs)
    if settrajs is not None:
        for d in rtrajs:
            if d['name'] in settrajs:
                setted.append(d)
                trajs.remove(d)
                num -= 1
    return setted + random.sample(trajs,num)

def selecttraj(trajpath,taskname, trajname):
    s = json.load(open(trajpath))
    trajs = s[taskname]['trajectories']
    for traj in trajs:
        if traj['name'] == trajname:
            return traj
    return None

def randominsertpos(trajectory):
    while True:
        i = random.randint(0,len(trajectory)-1)
        c = trajectory[i]
        if c[0] == 'option' and c[1][0][0:6] == 'SYSTEM':
            return i

def tasksampler(samplenum,trajpath,taskname=None,taskreq = None,settrajs = None):
    task = randomtraj(trajpath,taskname,1)[0]
    point = randominsertpos(task['subtasks_and_options'])
    if taskreq is not None:
        task = selecttraj(trajpath,taskname,taskreq)
        point = len(task['subtasks_and_options'])
    samples = randomtraj(trajpath, taskname, samplenum, exclude = task['name'],settrajs = settrajs)
    ret = []
    for sample in samples:
        ret.append(trajform(sample['subtasks_and_options']))
    return ret, trajform(task['subtasks_and_options'],point)

def getactionstatusprompt(historystr):
    demonstrations, task = tasksampler(5,'../datasets/SGD/trajectories/Hotels_1_trajectories.json')
    prompt = ''
    for demo in demonstrations:
        prompt += 'Demonstration: \n\n'+ demo + '\n'
    prompt += '\n'+'Based on the demonstrations above, predict the Actions and Statuses of each utterance in the following dialogs:\n\n'
    prompt += historystr
    prompt += '\n\nYour predictions for each utterance:'
    print(prompt)
    exit(0)
    return prompt

from GPT.OpenAIAPI import getcompletion
def getactionstatus(historystr):
    out = getcompletion(getactionstatusprompt(historystr))
    res = 'Actions: USER'+out
    res.replace('\n',' ')
    if 'Utterance' in res:
        res = res[0:res.index('Utterance')]
    return res



if __name__ == '__main__':
    #demonstrations, task = tasksampler(6,'../outputs/sample_data_n.json','Hotels_1_ReserveHotel','41_00048',['43_00078','41_00043','41_00034','41_00037','41_00044','41_00047'])
    trajs = randomtraj('../datasets/SGD/trajectories/Restaurants_1_trajectories.json',None,3)
    historystr = ''
    for i,traj in enumerate(trajs):
        historystr += 'Dialog ' + str(i+1)+':'
        historystr += trajform2(traj['subtasks_and_options'])
        historystr += '\n'
    getactionstatus(historystr)
        

    


        


            
