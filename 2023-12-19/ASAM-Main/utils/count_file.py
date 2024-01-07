import os

map=[]

dir = '/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/output/Inversion/SD-7.5-50-INV-5/embeddings'
for file in os.listdir(dir):
    id = int(file.split('_')[1])
    map.append(id)
    
for id in range(4000,8001):
    if id in map:
        pass
    else:
        print(id)
        
    