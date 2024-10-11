import json
import os
file = '../sam-1b/sa_000138-blip2-caption.json'
list = []

new_file = '../sam-1b/sa_000138-controlnet-validation.json'
new_f = open(new_file,'w')

with open(file,'r') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        line = json.loads(line)
        line['target'] = line['img']
        line['source'] = line['img'].replace('jpg','png')
        line.pop('img')
        list.append(line)
        json.dump(line,new_f)
        if i < len(lines)-1: new_f.write('\n')
        

