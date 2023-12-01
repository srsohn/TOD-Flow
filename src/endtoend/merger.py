import json, sys

# This script aggregates all results to be evaluated by MultiWOZ_Evaluation

namedict = {'hdsa':'hdsa','hdno':'hdno','galaxy':'galaxy-e2e','galaxystar':'galaxy-e2e'}

updated = json.load(open(sys.argv[3]+'/predictions/'+namedict[sys.argv[2].lower()]+'.json'))


for name in ['Attraction','Hotel','Train','Taxi','Restaurant','Attraction+Restaurant','Attraction+Restaurant+Taxi','Attraction+Hotel','Attraction+Taxi+Hotel','Attraction+Train','Restaurant+Hotel','Restaurant+Taxi+Hotel','Restaurant+Train','Hotel+Train']:
    a = json.load(open('../datasets/MultiWOZ/e2e/'+sys.argv[1]+'/'+name+'_select.json'))
    for b in a:
        for i in range(len(a[b])):
            updated[b][i] = a[b][i]

json.dump(updated, open('comboTest.json','w+'),indent=2)
