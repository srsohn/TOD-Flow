import os,sys
import json

# method name, one of HDMO, HDSA, GALAXY, GALAXYSTAR
method = sys.argv[1]
# Directory where all the can/shdnt graphs are stored
can_graph_dir = '../graphs/MultiWOZ/CSILP/'
# Directory where all the can/shdnt graphs are stored
shd_graph_dir = '../graphs/MultiWOZ/SHDILP/'
# Where your clone of MultiWOZ_Evaluation repo is
eval_repo_dir = '../../MultiWOZ_Evaluation'


cwd = os.getcwd()
# All possible hyperparameters
l1=['30.0','60.0','90.0','120.0','200.0','300.0'] # the POS for can/shdnt graph
l2=['0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.93','0.95'] # the mins for shd graph
l3=['1','2','10'] # The weight of can/shdnt violations versus shd violation
l4=['0'] # The thresholding of shd violations (i.e. we ignore violations under this number), we always use 0 (i.e. never ignore shd violations)

if method == 'GALAXY':
    l1 = ['30.0']
    l2 = ['0.8']
    l3 = ['1']
elif method == 'GALAXYSTAR':
    l1 = ['30.0']
    l2 = ['0.7']
    l3=['1']
elif method == 'HDNO':
    l1 = ['30.0']
    l2 = ['0.6']
    l3=['10']
elif method == 'HDSA':
    l1 = ['200.0']
    l2 = ['0.6']
    l3 = ['10']

resdict = {}
method2=method
if method == 'GALAXYSTAR':
    method2='GALAXY'
for a1 in l1:
    for a2 in l2:
        for a3 in l3:
            for a4 in l4:
                os.system('bash endtoend/generate_all_domains.sh '+method2+' '+a1+' '+a2+' '+a3+' '+a4+' '+can_graph_dir+' '+shd_graph_dir+' '+method+' '+eval_repo_dir+' > /dev/null')
                os.system('cd '+eval_repo_dir+' ; python3 evaluate.py --bleu --success  --input '+cwd+'/comboTest.json > /dev/null; cd '+cwd)
                eval_result = json.load(open(eval_repo_dir+'/evaluation_results.json'))
                bleu = eval_result['bleu']['mwz22']
                inf = eval_result['success']['inform']['total']
                suc = eval_result['success']['success']['total']
                s = ' '.join([a1,a2,a3,a4])
                resdict[s] = (bleu+(inf+suc)/2,bleu,inf,suc,s)
                print(resdict[s])

print(resdict)
json.dump(resdict,open('resdict.json','w+'))

l = []
for k in resdict:
    l.append(resdict[k])
l.sort(reverse=True)
print(l)
print('best result and hyperparameters: '+str(l[0]))



            
