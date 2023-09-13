import os
import shutil

dir1 = '/data/tanglv/Ad-SAM/2023-7-23/main/temp/skip-ablation-01-mi-0.5-sam-0.01-1-2-10-Clip-0.2/adv/'
dir2 = '/data/tanglv/Ad-SAM/2023-7-23/main/temp/skip-ablation-01-mi-0.5-sam-0.01-1-2-10-Clip-0.2/origin/'
dir3 = '/data/tanglv/data/sod_data/DUTS-TR/imgs/'

files = sorted(os.listdir(dir1))
for i in range(57):
    shutil.copy(dir3+files[i].replace('png','jpg'),dir2+files[i])

