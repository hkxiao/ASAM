import os
from pathlib import Path
import shutil

root = 'work_dirs'
methods = ['asam-cvpr-version','sam-baseline-vit_b']
datasets = ['Kvasir_SEG', 'Kvasir_sessile', 'CVC_ClinicDB']
save_root = root + '/choose_img_relative'
Path(save_root).mkdir(exist_ok=True, parents=True)

for dataset in datasets:
    
    info = []
    dataset_dir0 = os.path.join(root,methods[0], 'box', dataset)
    dataset_dir1 = os.path.join(root,methods[1], 'box', dataset)
    files0 = os.listdir(dataset_dir0)
    files0 = sorted([file for file in files0 if 'txt' in file])
    
    for file0 in files0:
        try:
            iou0 = float(open(os.path.join(dataset_dir0,file0),'r').read())
            iou1 = float(open(os.path.join(dataset_dir1,file0),'r').read())
            info.append((file0,iou0 - iou1))
        except:
            continue
    info = sorted(info, key=lambda x:x[1], reverse=True)
    
    save_dir = os.path.join(save_root, 'box' ,dataset)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    print(len(info))

    try:
        for i in range(10):
            shutil.copyfile(os.path.join(dataset_dir0,info[i][0].replace('.txt','.jpg')), os.path.join(save_dir,info[i][0].replace('.txt','_asam.jpg')))
            shutil.copyfile(os.path.join(dataset_dir0,info[i][0].replace('.txt','.jpg')), os.path.join(save_dir,info[i][0].replace('.txt','_asam.jpg')))
            shutil.copyfile(os.path.join(dataset_dir1,info[i][0].replace('.txt','.jpg')), os.path.join(save_dir,info[i][0].replace('.txt','_sam.jpg')))
            shutil.copyfile(os.path.join(dataset_dir0,info[i][0].replace('.txt','_gt.jpg')), os.path.join(save_dir,info[i][0]).replace('.txt','_gt.jpg'))
    except:
        continue
