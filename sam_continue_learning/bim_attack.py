import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from datetime import datetime
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
import torch.distributed as dist

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from torch.optim.lr_scheduler import LambdaLR,StepLR
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from easyrobust.easyrobust.third_party.vqgan import reconstruct_with_vqgan, VQModel
from PIL import Image
from pathlib import Path

def lr_lambda(epoch):
    if epoch < args.warmup_epoch:
        return (epoch + 1) / args.warmup_epoch  # warm up 阶段线性增加
    else:
        return args.gamma ** (epoch-args.warmup_epoch+1) # warm up 后每个 epoch 除以 2

def show_anns(labels_val, masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    print(masks.shape, len(ious), len(boundary_ious))
    for i, (label_val,mask, iou, biou) in enumerate(zip(labels_val, masks, ious, boundary_ious)):
       
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(label_val/255.0, plt.gca(), label_mode=True) 
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'_gt.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close() 
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box[i], plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point[i], input_label[i], plt.gca())
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close()
        
def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def show_mask(mask, ax, label_mode=False,random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif label_mode:
        color = np.array([122/255, 166/255, 82/255, 0.6])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
      
def record_iou(filename, ious, boundary_ious):
    if len(ious) == 0:
        return

    for i, (iou, biou) in enumerate(zip(ious, boundary_ious)):
        with open(filename+'_'+str(i)+'.txt','w') as f:
            f.write(str(round(iou.item()*100,2)))
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('Tune-SAM', add_help=False)

    # Base Setting
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--compile', action='store_true')
    
    parser.add_argument('--numworkers', type=int, default=-1)
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument("--restore-sam-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument('--train-datasets', nargs='+')
    parser.add_argument('--valid-datasets', nargs='+')
    parser.add_argument('--only_attack', action='store_true')
    parser.add_argument('--load_prefix', default='.', type=str)
    
    # SAM setting
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--baseline', action='store_true')
    
    # Tuning Decoder setting
    parser.add_argument('--tuning_part', default='output_token', choices=['output_token','decoder'])

    # Base Learning Rate Setting & Epoch
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--max_epoch_num', default=10, type=int)
    
    # Step Learning Rate
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    
    # Slow start & Fast decay
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--slow_start', action='store_true')
    
    # Input Setting
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_prompt_start', default=0, type=int)
    parser.add_argument('--batch_size_prompt', default=-1, type=int)
    parser.add_argument('--train_img_num', default=11186, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--prompt_type', default='box')
    parser.add_argument('--point_num', type=int, default=10)
    
    # DDP Setting
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    # Output Setting
    parser.add_argument("--output_prefix", type=str, required=False, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--model_save_fre', default=1, type=int)
    return parser.parse_args()

def FGSM(batched_input, target, model, clip_min, clip_max, eps=0.2):
    model.zero_grad()

    batched_output, interm_embeddings = model(batched_input, multimask_output=False)
    mask = batched_output[0]['low_res_logits']
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    loss = criterion(mask, target.detach())
    loss.backward()
    res = batched_input[0]['image'].grad

    ################################################################################
    batched_input[0]['image'] = batched_input[0]['image'] + eps * torch.sign(res)
    batched_input[0]['image'] = torch.max(batched_input[0]['image'], clip_min)
    batched_input[0]['image'] = torch.min(batched_input[0]['image'], clip_max)
    batched_input[0]['image'] = torch.clamp(batched_input[0]['image'], min=0.0, max=255.0)

    ################################################################################
    return batched_input

def BIM(model, batched_input, target, eps=0.03*255, k_number=2, alpha=0.01*255):
    batched_input[0]['ori_image'] = batched_input[0]['image'].detach().clone()
    image_unnorm = batched_input[0]['ori_image'].clone().detach()
    clip_min = image_unnorm - eps
    clip_max = image_unnorm + eps
    
    batched_input[0]['image'].requires_grad = True
    
    for mm in range(k_number):
        batched_input= FGSM(batched_input, target, model, clip_min, clip_max, eps=alpha)
        batched_input[0]['image'] = batched_input[0]['image'].detach()
        batched_input[0]['image'].requires_grad = True
        model.zero_grad()
        
    return batched_input

def main(train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        with open(args.output+'/log.txt','a') as f:
            f.write('\n\n\n=========>> '+str(datetime.now())+'\n')
            f.write(str(args)+'\n')
            
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        my_transforms = [Resize(args.input_size)] if args.only_attack else [RandomHFlip(),LargeScaleJitter()]
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = my_transforms,
                                                        batch_size = args.batch_size_train,
                                                        batch_size_prompt = args.batch_size_prompt,
                                                        batch_size_prompt_start = args.batch_size_prompt_start,
                                                        training = True,
                                                        numworkers=args.numworkers)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          batch_size_prompt = args.batch_size_prompt,
                                                          batch_size_prompt_start = args.batch_size_prompt_start,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    sam_checkpoint_map = {
        'vit_b': os.path.join(args.load_prefix,'pretrained_checkpoint/sam_vit_b_01ec64.pth'),
        'vit_l': '../pretrained_checkpoint/sam_vit_b_01ec64.pth',
        'vit_h': '../pretrained_checkpoint/sam_vit_b_01ec64.pth',
    }
    sam = sam_model_registry[args.model_type](sam_checkpoint_map[args.model_type])
    if args.compile: sam = torch.compile(sam)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    sam_without_ddp = sam.module
    
    parameters_grad, parameters_no_grad = 0, 0
    for n,p in sam.named_parameters():
        if p.requires_grad: parameters_grad += 1
        else: parameters_no_grad += 1
    print("parameters_grad Number:", parameters_grad)
    print("parameters_no_grad Number:", parameters_no_grad)
 
    if args.restore_model:
        print("restore model from:", args.restore_model)
        if torch.cuda.is_available():
            sam_without_ddp.load_state_dict(torch.load(args.restore_model,map_location='cuda'))
        else:
            sam_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
            
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(sam_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        if not args.slow_start:
            lr_scheduler = StepLR(optimizer, args.lr_drop_epoch, last_epoch=args.start_epoch)
        else:
            print("slow start & fast decay")
            lr_scheduler = LambdaLR(optimizer, lr_lambda)

        train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:    
        evaluate(args, sam, valid_dataloaders, args.visualize)


def train(args, sam, optimizer, train_dataloaders, vaid_dataloaders, lr_scheduler):
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    
    Path(os.path.join(args.output,'adv_examples',train_datasets[0]['name'])).mkdir(exist_ok=True, parents=True)    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
 
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
    
        for data in metric_logger.log_every(train_dataloaders,10):
            
            inputs, labels, ori_im_path = data['image'], data['label'], data['ori_im_path']  # [K 3 1024 1024]   [K N 1024 1024]
            K, N, H, W =labels.shape
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
                labels = labels.reshape(K*N,H,W).cuda()  #[K*N 1024 1024]
                
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()  #[K 1024 1024 3]
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels) #[K*N 4]
            try:
                labels_points = misc.masks_sample_points(labels) #[K*N 10 2]
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear')
            imgs_256 = F.interpolate(inputs, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256) #[K*N 1 256 256]

            batched_input = []
            
            for bi in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[bi].astype(dtype=np.float32), device=sam.device).permute(2, 0, 1).contiguous() # [3 1024 1024]
                dict_input['image'] = input_image.to(torch.float32) 
                
                input_type = random.choice(input_keys)
                sparse_slice, dense_slice = slice(bi*N,(bi+1)*N), slice(bi*N,(bi+1)*N)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[sparse_slice,...]  #N*4
                elif input_type == 'point':
                    point_coords = labels_points[sparse_slice,...] # N 10 2
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[:-1], device=point_coords.device) #[N 10]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[dense_slice] # N 1 256 256

                else:
                    raise NotImplementedError

                dict_input['original_size'] = imgs[0].shape[:2]
                batched_input.append(dict_input)

            advinput = BIM(sam, batched_input, labels_256)
            img_np = advinput[0]['image'].permute(1,2,0).cpu().data.numpy()
            cv2.imwrite(os.path.join(args.output,'adv_examples',train_datasets[0]['name'],ori_im_path[0].split('/')[-1]), img_np[:,:,::-1].astype(np.uint8))

    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
@torch.no_grad()
def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    iou_list = []
    for i in range(0,len(preds)):
        single_iou = misc.mask_iou(postprocess_preds[i],target[i])
        iou = iou + single_iou
        iou_list.append(single_iou)
    return iou / len(preds), iou_list

@torch.no_grad()
def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    iou_list = []
    for i in range(0,len(preds)):
        single_iou = misc.boundary_iou(target[i],postprocess_preds[i])
        iou = iou + single_iou
        iou_list.append(single_iou)
    return iou / len(preds), iou_list


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------
    dataset_sa000000 = {"name": "sam_subset",
                "im_dir": "../../sam-1b/sa_000000",
                "gt_dir": "../../sam-1b/sa_000000",
                "im_ext": ".jpg",
                "gt_ext": ""}
    
    dataset_sa000000_512 = {"name": "sam_subset",
            "im_dir": "../../sam-1b/sa_000000/512",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".jpg",
            "gt_ext": ""}
    
    dataset_sa000000adv = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-100-1-2-10-Clip-0.2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sa000000adv_dice = {"name": "sam_subset",
        "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000_Inversion = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Inversion/inv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sa000000adv_1600 = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-1600.0-1-2-10-Clip-0.2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sa000000inv = {"name": "sam_subset",
        "im_dir": "../../output/sa_000000@4-Grad/diversity-01-mi-SD-9.0-20-SAM-sam-vit_b-4-ADV-0.2-10-0.02-0.5-10.0-0.1-2/adv",
        "gt_dir": "/data/tanglv/data/sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
            
    dataset_sam_subset_adv_vit_huge = {"name": "sam_subset",
            "im_dir": "../../11187-Grad/skip-ablation-01-mi-0.5-sam-vit_h-40-0.01-100-1-2-10-Clip-0.2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ".json"}
    
    dataset_DatasetDM = {"name": "DatasetDM",
            "im_dir": "../../data/tanglv/xhk/DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/Image",
            "gt_dir": "../../data/tanglv/xhk/DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/label",
            "im_ext": ".jpg",
            "gt_ext": ".jpg"}
    
    dataset_sa000000pgd = {"name": "sam_subset",
            "im_dir": "../../data/tanglv/xhk/Ad-Sam-Main/sam_continue_learning/train/work_dirs/PGD",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".jpg",
            "gt_ext": ".json"}
    
    dataset_sa00000pgd_512 = {"name": "sam_subset",
        "im_dir": "work_dirs/PGD_512",
        "gt_dir": "../../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    # valid set
    
    # single
    dataset_hrsod_val = {"name": "HRSOD-TE",
            "im_dir": "data/HRSOD-TE/imgs",
            "gt_dir": "data/HRSOD-TE/gts",
            "im_ext": ".jpg",
            "gt_ext": ".png"}

    #全景分割
    dataset_ade20k_val = {"name": "ADE20K_2016_07_26",
            "im_dir": "../data/ADE20K_2016_07_26/images/validation",
            "gt_dir": "../data/ADE20K_2016_07_26/images/validation",
            "im_ext": ".jpg",
            "gt_ext": "_seg.png"}
    #实列分割
    dataset_cityscapes_val = {"name": "cityscaps_val",
            "im_dir": "data/cityscapes/leftImg8bit/val",
            "gt_dir": "data/cityscapes/gtFine/val",
            "im_ext": "_leftImg8bit.png",
            "gt_ext": "_gtFine_instanceIds.png"}
    #实列分割
    dataset_voc2012_val = {"name": "voc2012_val",
            "im_dir": "data/VOC2012/JPEGImages_val",
            "gt_dir": "data/VOC2012/SegmentationObject",
            "im_ext": ".jpg",
            "gt_ext": ".png"}
    #实列分割
    dataset_coco2017_val = {"name": "coco2017_val",
            "im_dir": "../data/COCO2017-val/val2017",
            "annotation_file": "../data/COCO2017-val/instances_val2017.json",
            "im_ext": ".jpg"
            }
    dataset_camo = {"name": "camo",
        "im_dir": "../data/CAMO/imgs",
        "gt_dir": "../data/CAMO/gts",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ishape_antenna = {"name": "ishape",
        "im_dir": "../data/ishape_dataset/antenna/val/image",
        "gt_dir": "../data/ishape_dataset/antenna/val/instance_map",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "../data/Plant_Phenotyping_Datasets",
        "gt_dir": "../data/Plant_Phenotyping_Datasets",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
        }
    
    dataset_gtea_train = {"name": "gtea",
            "im_dir": "../data/GTEA_hand2k/GTEA_GAZE_PLUS/Images",
            "gt_dir": "../data/GTEA_hand2k/GTEA_GAZE_PLUS/Masks",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        }
    
    dataset_streets = {"name": "streets_coco",
        "im_dir": "../data/vehicleannotations/images",
        "annotation_file": "../data/vehicleannotations/annotations/vehicle-annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_TimberSeg = {"name": "timberseg_coco",
        "im_dir": "..//data/y5npsm3gkj-2/prescaled/",
        "annotation_file": "../data/y5npsm3gkj-2/prescaled/coco_annotation_rotated.json",
        "im_ext": ".png",
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "../data/Plant_Phenotyping_Datasets",
        "gt_dir": "../data/Plant_Phenotyping_Datasets",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
        }
    
    dataset_gtea_train = {"name": "gtea",
        "im_dir": "../data/GTEA_GAZE_PLUS/Images",
        "gt_dir": "../data/GTEA_GAZE_PLUS/Masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_streets = {"name": "streets_coco",
        "im_dir": "../data/vehicleannotations/images",
        "annotation_file": "../data/vehicleannotations/annotations/vehicle-annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_big_val = {"name": "big",
        "im_dir": "../data/BIG/val",
        "gt_dir": "../data/BIG/val",
        "im_ext": "_im.jpg",
        "gt_ext": "_gt.png"
    }
    
    dataset_ndis_train = {"name": "ndis_park_coco",
        "im_dir": "../data/ndis_park/train/imgs",
        "annotation_file": "../data/ndis_park/train/train_coco_annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_test = {"name": "Plittersdorf_coco",
        "im_dir": "../data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "../data/plittersdorf_instance_segmentation_coco/test.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_train = {"name": "Plittersdorf_coco",
        "im_dir": "../data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "../data/plittersdorf_instance_segmentation_coco/train.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_val = {"name": "Plittersdorf_coco",
        "im_dir": "../data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "../data/plittersdorf_instance_segmentation_coco/val.json",
        "im_ext": ".jpg",
    }
    
        
    dataset_egohos = {"name": "egohos",
        "im_dir": "../data/egohos/val/image",
        "gt_dir": "../data/egohos/val/label",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_LVIS = {"name": "LVIS",
        "im_dir": "../data/LVIS/val2017",
        "annotation_file": "../data/LVIS/annotations/lvis_v1_val.json",
        "im_ext": ".jpg",
    }
    dataset_BBC038v1 = {"name": "BBC038v1",
        "im_dir": "../data/BBC038V1-Train",
        "annotation_file": "../data/BBC038V1-Train",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS1 = {"name": "DOORS1",
        "im_dir": "../data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/img",
        "gt_dir": "../data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS2 = {"name": "DOORS2",
        "im_dir": "../data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/img",
        "gt_dir": "../data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    
    dataset_NDD20_ABOVE = {"name": "NDD20_coco",
        "im_dir": "../data/NDD20/ABOVE",
        "annotation_file": "../data/NDD20/ABOVE_LABELS.json",
        "im_ext": ".jpg",
    }
    
    
    dataset_ZeroWaste = {"name": "ZeroWaste",
        "im_dir": "../data/splits_final_deblurred/train/data",
        "gt_dir": "../data/splits_final_deblurred/train/sem_seg",
        "im_ext": ".PNG",
        "gt_ext": ".PNG"
    }
    
    
    args = get_args_parser()
    if not args.eval:
        args.output = os.path.join('work_dirs', args.output_prefix+'-'+args.train_datasets[0].split('_')[-1]+'-'+args.model_type)
    elif args.baseline:
        args.output = os.path.join('work_dirs', args.output_prefix+'-'+args.model_type)
    else:
        args.output = os.path.join(*args.restore_model.split('/')[:-1])
        
    train_datasets = [globals()[dataset] for dataset in args.train_datasets]
    valid_datasets = [globals()[dataset] for dataset in args.valid_datasets]

    for train_dataset in train_datasets:
        train_dataset['im_dir'] = os.path.join(args.load_prefix, train_dataset['im_dir'])
        if 'gt_dir' in train_dataset: train_dataset['gt_dir'] = os.path.join(args.load_prefix, train_dataset['gt_dir'])
        if 'annotation_file' in train_dataset: train_dataset['annotation_file'] = os.path.join(args.load_prefix, train_dataset['annotation_file'])
        
    for test_dataset in valid_datasets:
        test_dataset['im_dir'] = os.path.join(args.load_prefix, test_dataset['im_dir'])
        if 'gt_dir' in test_dataset: test_dataset['gt_dir'] = os.path.join(args.load_prefix, test_dataset['gt_dir'])
        if 'annotation_file' in test_dataset: test_dataset['annotation_file'] = os.path.join(args.load_prefix, test_dataset['annotation_file'])
        
    main(train_datasets, valid_datasets, args)
