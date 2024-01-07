import os
import shutil

dir = '/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/output/Inversion/SD-7.5-50-INV-5/embeddings/'
dir2 = '/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/output/Inversion/SD-7.5-50-INV-5/embeddings2/'

for file in os.listdir(dir):
    id = float(file.split("_")[1])
    if id >= 4000 and id <=8000:
        shutil.move(dir+file, dir2+file)
        
        
