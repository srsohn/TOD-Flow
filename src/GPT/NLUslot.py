import json
import random
import copy
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
            if c[1][0] == 'SYSTEM_Query':
                continue
            res += 'Utterance: '+c[2]+'\n'
            newstrs = [a+' : '+c[3][a] for a in c[3]]
            res += 'Slot Updates: ' + ' | '.join(newstrs)+'\n'

    return res

def randomtraj(trajpath,taskname, num, exclude = None,settrajs = None):
    s = json.load(open(trajpath))
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

def tasksampler(samplenum,trajpath,taskname,taskreq = None,settrajs = None):
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

def getslotupdateprompt(historystr):
    demonstrations, task = tasksampler(7,'../outputs/sample_data_n.json','Hotels_1_ReserveHotel','41_00048',['43_00078','41_00043','41_00034','41_00037','41_00044','41_00047','41_00013'])
    prompt = ''
    for demo in demonstrations:
        prompt += 'Demonstration: \n\n '+ demo + '\n'
    prompt += '\n'+'Based on the demonstrations above, list all slot updates in the utterance below. Only include updated in the utterance below. Do not make up information that does not exist.\n\n'
    prompt += historystr
    prompt += '\n Slot Updates:'
    return prompt

from GPT.OpenAIAPI import getcompletion
def getslotupdate(historystr,verbose=False):
    res = getcompletion(getslotupdateprompt(historystr))
    if verbose:
        print(res)
    res.replace('\n',' ')
    if 'Utterance' in res:
        res = res[0:res.index('Utterance')]
    return parseresult(res)

def spaceremoval(s):
    if len(s) == 0:
        return s
    if s[0] == ' ':
        return spaceremoval(s[1:])
    if s[-1] == ' ':
        return spaceremoval(s[0:-1])
    return s

def parseresult(res):
    reses = res.split('|')
    ret = {}
    for rs in reses:
        rss = rs.split(':')
        if len(rss) < 2:
            return ret
        if len(rss[1]) > 0:
            ret[spaceremoval(rss[0])] = spaceremoval(rss[1])

    return ret

if __name__ == '__main__':
    demonstrations, task = tasksampler(6,'../outputs/sample_data_3.json','Hotels_1_ReserveHotel','41_00048',['43_00078','41_00043','41_00034','41_00037','41_00044','41_00047','41_00013'])
    for d in demonstrations:
        print('Demonstration: \n')
        print(d)
        print('\n')

    print('Based on the demonstrations above, predict the Action and Status of the utterance: \n')
    print(task)
    print('Actions: ')


        


            
