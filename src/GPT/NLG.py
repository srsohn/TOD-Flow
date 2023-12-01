from GPT.GPTNeobaseline import rs
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
rss = [rs() for i in range(500)]
import random
def getact(sent):
    li = sent.rindex('Actions: ')
    return sent[li+9:-2]

acts = [getact(s[0]) for s in rss]
embeds = model.encode(acts)
from sklearn.metrics.pairwise import cosine_similarity

def retrieveclosest(a,num=5):
    embed = model.encode([a])
    cossims = [(1-cosine_similarity(embedd.reshape((1,768)), embed),i) for i,embedd in enumerate(embeds)]
    cossims.sort()
    rets = [rss[id][1] for _,id in cossims[0:num]] + [a for _,a in random.sample(rss,25)]
    return rets

def createslotstr(slots):
    sl = [(a,slots[a]) for a in slots]
    slotsstr = ""
    for a,b in sl:
        slotsstr += a + ' : ' + b + ' | '
    slotsstr = slotsstr[:-3]
    return 'Slots: '+slotsstr

def getnlgprompt(slots,actionstatus):
    actions = actionstatus[9:actionstatus.index(';')-1]
    demonstrations = retrieveclosest(actions)
    ret = 'Demonstrations:\n'
    for d in demonstrations: 
        ret += '\n'+d+'\n'
    ret += '\nBased on the demonstrations above, make the prediction below. You should only perform the actions listed and nothing else. Slots provided may be redundant and you do not need to use all of them. \n\n'
    ret += createslotstr(slots)+' ; '
    ret += actions +' ; Predictions:'
    return ret

from GPT.OpenAIAPI import getcompletion
def getnlg(slots,actionstatus):
    prompt = getnlgprompt(slots,actionstatus)
    return getcompletion(prompt)[1:]





