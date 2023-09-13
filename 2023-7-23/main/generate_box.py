import torch
from PIL import Image
import os
from tqdm import tqdm, trange
import json
import torch
from lavis.models import load_model_and_preprocess
import cv2
from PIL import Image
import numpy as np

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

dir = '/data/tanglv/data/sod_data'
datasets = ['DUTS-TR']

# loads BLIP-2 pre-trained model
#model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)

def show_box(img,s_p,e_p):
    color = (0, 255, 0)  # 绿色
    image_with_rectangle = cv2.rectangle(img.permute(1,2,0).numpy(), s_p, e_p, color, thickness=2)
    cv2.imwrite('demo_box.png',image_with_rectangle)
    
for dataset in datasets:
    dataset_dir = os.path.join(dir, dataset)
    imgs_dir = os.path.join(dir, dataset, 'imgs')
    gts_dir = os.path.join(dir, dataset, 'gts')
    
    prompt_path = dataset_dir+'/box.json'
    if os.path.exists(prompt_path):
        os.remove(prompt_path)
    
    f = open(prompt_path,'a')
    
    img_files,gt_files = sorted(os.listdir(imgs_dir)),sorted(os.listdir(gts_dir))
        
    all_info = open(os.path.join(dataset_dir,'blip2_prompt.json')).readlines()
    
    for i in trange(len(img_files)):    
        info = json.loads(all_info[i])
        # prepare the image
        img_file,gt_file = info['target'].split('/')[-1], info['source'].split('/')[-1]
        print(img_file)
        example = {}
        example["source"] = "gts/" + gt_file
        example["target"] = "imgs/" + img_file
        
        gt_path = os.path.join(gts_dir,gt_file)
        gt = Image.open(gt_path).resize((1024,1024))
        gt = torch.from_numpy(np.array(gt)).permute(2,0,1)
        
        index = torch.nonzero(gt) # N * 3
        u,d, l,r = index[:,1].min(), index[:,1].max(),index[:,2].min(), index[:,2].max()
        u,d, l,r = u.item(),d.item(), l.item(),r.item()
    
        # show_box(gt,(l,u),(r,d))
        # raise NameError    
        example["box"] = [l,u,r,d]
        
        json.dump(example,f)
      
        #prompt = example["prompt"]
        
        f.write(   f"\n"     )
        #f.write(   f"{prompt}\n"     )
    
    f.close()