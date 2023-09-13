import torch
from PIL import Image
import os
from tqdm import tqdm, trange
import json

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

import torch
from lavis.models import load_model_and_preprocess

dir = '/data/tanglv/data/sod_data/'
datasets = ['DUTS-TR']

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)

for dataset in datasets:
    dataset_dir = os.path.join(dir, dataset)
    imgs_dir = os.path.join(dir, dataset, 'imgs')
    gts_dir = os.path.join(dir, dataset, 'gts')
    
    prompt_path = dataset_dir+'/blip2_prompt.json'
    if os.path.exists(prompt_path):
        os.remove(prompt_path)
    
    f = open(prompt_path,'a')
    
    
    img_files,gt_files = sorted(os.listdir(imgs_dir)),sorted(os.listdir(gts_dir))

    # print(img_files, gt_files)
    #for img_file,gt_file in tqdm(zip(img_files, gt_files)): 
        
    for i in trange(len(img_files)):    
        # prepare the image
        img_file,gt_file = img_files[i], gt_files[i]
        print(img_file)
        raw_image = Image.open(os.path.join(imgs_dir,img_file)).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        example = {}
        example["source"] = "gts/" + gt_file
        example["target"] = "imgs/" + img_file
        example["prompt"] = model.generate({"image": image})[0]
        
        json.dump(example,f)
        
        f.write(   f"\n"     )
    
    f.close()