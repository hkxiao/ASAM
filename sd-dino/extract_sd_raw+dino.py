import os
import torch
torch.set_num_threads(16)
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from extractor_dino import ViTExtractor
from extractor_sd import load_model, process_features_and_mask
import torch.nn.functional as F
import json
from pathlib import Path

def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def main(args):
    # random setting
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    
    # load stable diffusion
    ldm, aug = load_model(diffusion_ver=args.VER, image_size=args.SIZE, num_timesteps=args.TIMESTEP, block_indices=tuple(args.INDICES))
    real_size=960
    
    # load DINO v2
    MODEL_SIZE = args.MODEL_SIZE
    img_size = 840 
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    model_type = model_dict[MODEL_SIZE] 
    layer = 11 
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' 
    stride = 14
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor_dino = ViTExtractor(model_type, stride, device=device)
    
    # presolve imgs & caption
    img_files = os.listdir(args.img_dir)
    img_files = [file for file in img_files if 'jpg' in file]
    captions = {}
    with open(args.caption_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dict = json.loads(line.strip()) 
            captions[json_dict['img'].strip()] = json_dict['prompt'].strip()   

    # feature extract loop
    #N = len(img_files) 
    pbar = tqdm(total=args.end-args.start)

    for idx in range(args.start,args.end):
        # load image 
        img1 = Image.open(os.path.join(args.img_dir ,img_files[idx])).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=False) # SD input
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)  # DINO input

        # load text prompot 
        if args.TEXT_INPUT: input_text = captions[img_files[idx]]
        else: input_text = ""
        
        # extract feature
        with torch.no_grad():
            img1_desc = process_features_and_mask(ldm, aug, img1_input, input_text=input_text, mask=False, raw=True)
            img1_batch = extractor_dino.preprocess_pil(img1)
            img1_desc_dino = extractor_dino.extract_descriptors(img1_batch.to(device), layer, facet)
            
            state_dict = {}
            state_dict['sd_feat'] = img1_desc
            state_dict['dino_feat'] = img1_desc_dino            
            
            torch.save(img1_desc, args.output_dir + '/' + img_files[idx].replace('.jpg','_sd.pth'))
            torch.save(img1_desc_dino, args.output_dir + '/' + img_files[idx].replace('.jpg','_dino.pth'))
            #torch.save(state_dict,files[idx].replace('imgs','sd_raw+dino_feat')[:-4]+'.pth')

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7 python extract_sd_raw+dino.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--output_dir', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--caption_path', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--start', type=int, default=42)
    parser.add_argument('--end', type=int, default=42)
    
    # Stable Diffusion Setting
    parser.add_argument('--VER', type=str, default="v1-5")                          # version of diffusion, v1-3, v1-4, v1-5, v2-1-base
    parser.add_argument('--PROJ_LAYER', action='store_true', default=False)         # set true to use the pretrained projection layer from ODISE for dimension reduction
    parser.add_argument('--SIZE', type=int, default=960)                            # image size for the sd input
    parser.add_argument('--INDICES', nargs=4, type=int, default=[2,5,8,11])         # select different layers of sd features, only the first three are used by default
    parser.add_argument('--WEIGHT', nargs=5, type=float, default=[1,1,1,1,1])       # first three corresponde to three layers for the sd features, and the last two for the ensembled sd/dino features
    parser.add_argument('--TIMESTEP', type=int, default=100)                        # timestep for diffusion, [0, 1000], 0 for no noise added

    # DINO Setting
    parser.add_argument('--MODEL_SIZE', type=str, default='base')                   # model size of thye dinov2, small, base, large
    parser.add_argument('--TEXT_INPUT', action='store_true', default=False)         # set true to use the explicit text input
    parser.add_argument('--NOTE', type=str, default='')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    main(args)