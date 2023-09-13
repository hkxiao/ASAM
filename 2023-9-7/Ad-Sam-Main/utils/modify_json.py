import json
import os
file = 'sam-1b-controlnet-train.json'
list = []

new_file = 'sam-1b-controlnet-train_2.json'
new_f = open(new_file,'w')

with open(file,'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        line['target'] = line['img']
        line['source'] = line['img'][:-4]+'.png'
        line.pop('img')
        list.append(line)
        json.dump(line,new_f)
        new_f.write('\n')
        

