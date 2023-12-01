import json
import sys

namewoz = sys.argv[1]
a = json.load(open(namewoz.lower()+'tmpTest.json'))

def removelines(s):
    if len(s)==0:
        return s
    if s[0] == '\n':
        return removelines(s[1:])
    if s[-1] == '\n':
        return removelines(s[:-1])
    return s

for name in a:
    for i in range(len(a[name])):
        s = removelines(a[name][i])
        lines = s.split('\n')
        if len(lines) != 7:
            print(name+' '+str(i))
            continue
        for ii in range(7):
            if '('+str(ii+1)+')' in lines[ii]:
                lines[ii] = lines[ii][lines[ii].index('('+str(ii+1)+')'):]
            if '(' ==lines[ii][0]:
                lines[ii] = lines[ii][4:]
                if len(lines[ii]) > 0 and lines[ii][0] == ' ':
                    lines[ii] = lines[ii][1:]
            if ': ' in lines[ii]:
                lines[ii] = lines[ii][lines[ii].index(': ')+2:]
            splitted = lines[ii].split(' | ')
            lines[ii] = splitted 
        a[name][i] = lines

json.dump(a,open('../datasets/MultiWOZ_full/HDNOActions/'+namewoz+'_actions.json','w+'),indent=2)
        
