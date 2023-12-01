import json
import sys
namewoz = sys.argv[1]
a = json.load(open('../datasets/MultiWOZ/e2e/'+sys.argv[2]+'/'+namewoz+'_actions.json'))
alls = []
for name in a:
    turns = []
    d = {'dialog_id':name,'turns':turns}
    for i in range(len(a[name])):
        acts = [[],[],[],[],[],[],[]]
        turn = {'turn_id':i,'USER':{'action':[],'status':[]},'SYSTEM':{'action':acts,'status':[]}}
        if a[name][i] is not None and len(a[name][i]) == len(acts):
            for j in range(len(acts)):
                acts[j] = a[name][i][j]
        turns.append(turn)
    alls.append(d)

json.dump(alls,open('../datasets/MultiWOZ/e2e/'+sys.argv[2]+'/'+namewoz+'_predform.json','w+'),indent=2)
        
