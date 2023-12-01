import json

# post-processing on the actions/statuses
def postprocess(li):
    ret = []
    for s in li:
        if 'inform none' in s.lower():
            continue
        ret.append(s.lower().replace('select','ask-for-selection').replace(' none','').replace('choice','num_choices').replace('user','USER').replace('system','SYSTEM'))
    if len(ret) == 0:
        ret = ['NONE']
    ret = list(dict.fromkeys(ret)) # deduplicate
    return ret

# converts a list of actions/statuses into a list of indices from an "all" list that contains all the actions/statuses
def listtoindex(li,all):
    return [all.index(x) for x in li]

# the main function that post-processes the multiwoz dataset that supports multi-domains
def preprocess_multiwoz(datajson,trainlist,devlist,testlist,domain,namelist,outjson):
    """
    datajson: the path to a json file that contains the multiwoz data
    trainlist: list of ids of training dialogs
    devlist: list of ids of dev dialogs
    testlist: list of ids of test dialogs
    domain: the domain to obtain trajectories for
    namelist: the list of ids of dialogs that belongs to this domain
    outjson: the json file to output the processed trajectories to
    """
    datas = json.load(open(datajson))
    allacts = []
    allstatuses = []
    alltrajs = []
    domains = []
    nums = [0,0,0]
    counter = 0
    out = {'option_labels':allacts,'subtask_labels':allstatuses,'num_subtask':len(allstatuses),'num_option':len(allacts),'trajectories':alltrajs}
    for name in datas:
        if name.lower() not in delexjson:
            print('warning: not found in delex '+name)
            continue
        if name+'_0' not in namelist:
            continue
        dialog = datas[name]
        possible = []
        for g in dialog['new_goal']:
            possible.append(g)
        
        dialogmulti = [(len(dialog['log']),None)]
        startpoint = 0
        if name[0:3] in ['SNG','SSN','WOZ'] and len(dialogmulti) > 1:
            # we print a warning if we find actions from multiple domains in a supposed single-domain dialog
            print('Warning '+name)
        for idx,multi in enumerate(dialogmulti):
            endpt,dom = multi

            # a few dialogs have messed up endings, we only use the non-messed-up parts
            if name == 'SNG1213.json':
                endpt = 12
            if name == 'PMUL0237.json':
                endpt = 29
            if name == 'PMUL0382.json':
                endpt = 21

            actslist = []
            for turn in dialog['log'][startpoint:endpt]:
                speaker = 'user'
                if len(turn['metadata']) > 0:
                    speaker = 'system'
                acts = []
                bookstatus = None
                offerstatus = True
                for actname in turn['dialog_act']:
                    for lis in turn['dialog_act'][actname]:
                        acnames = actname.split('-')
                        if acnames[1] == 'Recommend':
                            acnames[1] = 'Inform'
                        if acnames[1] == 'NoOffer':
                            acnames[1] == 'no-offer'
                            offerstatus = False
                            acts.append(speaker+' '+acnames[0]+'-'+acnames[1])
                        elif actname == 'Booking-Inform':
                            if speaker == 'system':
                                acnames[1] = 'OfferBook'
                            else:
                                acnames[1] = 'Want-to-book'
                            acts.append(speaker+' '+acnames[0]+'-'+acnames[1])
                            acts.append(speaker+' '+acnames[0]+'-Inform '+lis[0])
                        elif acnames[0] == 'Booking' and acnames[1] != 'Request':
                            bookstatus = (acnames[1] == 'Book')
                            acts.append('system '+acnames[0]+'-'+acnames[1])
                            if bookstatus:
                                acts.append('system '+acnames[0]+'-Inform '+lis[0])
                        else:
                            acts.append(speaker+' '+acnames[0]+'-'+acnames[1]+' '+lis[0])
                acts = postprocess(acts)
                for act in acts:
                    if act not in allacts:
                        allacts.append(act)
                actslist.append([acts,turn['text'],bookstatus,offerstatus,speaker])
            for i in range(len(actslist)):
                a,b,c,o,d = actslist[i]
                statuses = []
                for act in a:
                    statuses.append('status '+act)
                if d == 'user' and i < len(actslist)-1:
                    _,_,cc,oo,_ = actslist[i+1]
                    if cc is not None:
                        if cc:
                            statuses.append('status can book')
                        else:
                            statuses.append('status cannot book')
                    if not oo:
                        statuses.append('status no available offer')
                for s in statuses:
                    if s not in allstatuses:
                        allstatuses.append(s)
                actslist[i].append(statuses)
            if name in trainlist:
                split='train'
                nums[0] += 1
            elif name in devlist:
                split='val'
                nums[1] += 1
            elif name in testlist:
                split='test'
                nums[2] += 1
            else:
                print(name+' not in train, val or test set')
                continue
            so=[]
            soindices=[]
            traj = {'name':name, 'split' : split,'subtasks_and_options':so,'subtask_and_option_indices':soindices}
            if dom not in domains:
                domains.append(dom)
            for i,(o,utter,_,_,speaker,s) in enumerate(actslist):
                delexed = getdelex(name,startpoint + i)
                if delexed is None:
                    continue
                so.append(['option',o,utter,delexed])
                so.append(['subtask',s])
                soindices.append(['option',listtoindex(o,allacts)])
                soindices.append(['subtask',listtoindex(s,allstatuses)])
            alltrajs.append(traj)
            startpoint = endpt
    print(len(alltrajs))
    print(nums)
    print(counter)
    f=open(outjson,'w+')

    json.dump({domain:out},f,indent=2)


# obtain delexicalized dialog utterances for E2E evaluation purposes
delexjson = json.load(open('../datasets/MultiWOZ/data_for_galaxy.json'))
def getdelex(name,num):
    delexdict = delexjson[name.lower()]
    tn = num // 2
    speaker = 'user_delex'
    if num > tn * 2:
        speaker = 'resp'
    try:
        return delexdict['log'][tn][speaker]
    except:
        print("warning")
        print(name)
        print(num)
        return None

import torch
if __name__=='__main__':
    # load list of dialog ids from each split
    trainlist,vallist,testlist = torch.load('../datasets/MultiWOZ/splitlists.pt')
    # load list of dialog ids from each domain combination
    namelists = json.load(open('../datasets/MultiWOZ/Domains.json'))
    for domain in namelists:
        if domain in ['police','hospital']:
            continue
        nl3 = namelists[domain]
        namelist = nl3['train']+nl3['val']+nl3['test']
        preprocess_multiwoz('../../MultiWOZ2_3/data.json',trainlist,vallist,testlist,domain.lower(),namelist,'../datasets/MultiWOZ/trajectories/'+domain+'_trajectories.json')







        
    
    
    
        
    

    

        




