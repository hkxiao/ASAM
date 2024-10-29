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
from torch.optim.lr_scheduler import LambdaLR, StepLR
from pathlib import Path
import sys
import copy
import sys
sys.path.append('sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
sys.path.append('SAMed')
#from SAMed.sam_lora_image_encoder import LoRA_Sam
import yaml
# sys.path.append('SAM-Adapter-PyTorch')
# import models
import copy


def print_current_line(current_frame):
    # 获取当前行号
    line_number = current_frame.f_lineno
    # 打印当前行号
    print(f"当前错误的行号是: {line_number}")
    
def lr_lambda(epoch):
    if epoch < args.warmup_epoch:
        return (epoch + 1) / args.warmup_epoch  # warm up 阶段线性增加
    else:
        return args.gamma ** (epoch-args.warmup_epoch+1) # warm up 后每个 epoch 除以 2

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoder_Tuning(MaskDecoder):
    def __init__(self, model_type, tuning_manner='output_tokens'):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256)
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_eval_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b": os.path.join(args.load_prefix,"../pretrained/sam_vit_b_maskdecoder.pth"),
                           "vit_l":"../pretrained/sam_vit_l_maskdecoder.pth",
                           'vit_h':"../pretrained/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("Tune Decoder init from SAM MaskDecoder")
        
        if tuning_manner == 'output_tokens':
            for n,p in self.named_parameters():
                if 'mask_tokens' not in n and 'iou_token' not in n:
                    p.requires_grad = False
                else :
                    print('Second Stege MaksDecoder:', n, 'need gradient', p.shape)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          eval_multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """


        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)
        
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) #[5 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[N 5 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred



def show_anns(labels_val, masks, input_point, input_label, input_box, filename, image, ious, boundary_ious, save_gt=True, suffix="", boxes_color=""):
    if len(masks) == 0: return
    
    #import pdb; pdb.set_trace()
    for i, (label_val,mask, iou, biou) in enumerate(zip(labels_val, masks, ious, boundary_ious)):        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(label_val/255.0, plt.gca(), label_mode=True) 
        plt.axis('off')
        if save_gt: plt.savefig(filename+'_'+str(i)+'_gt.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close() 
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box[i], plt.gca(), color=boxes_color)
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point[i], input_label[i], plt.gca())
        plt.axis('off')
        
        #print(filename+'_'+str(i)+'.jpg',)
        if suffix=="": plt.savefig(filename+'_'+str(i)+'.jpg',bbox_inches='tight',pad_inches=-0.1)
        else: plt.savefig(filename+'_'+str(i)+'_'+suffix+'.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_anns2(labels_val, masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return
    
    gt_mask_show, pred_mask_show = np.zeros((1024,1024,3)), np.zeros((1024,1024,3))
    length = 1<<24
    num = labels_val.shape[0]
    for i, (label_val,mask, iou, biou) in enumerate(zip(labels_val, masks, ious, boundary_ious)):
        
        label_val = (label_val>=123.).to(torch.int).detach().cpu().squeeze().numpy()
        mask = (mask>=0.5).to(torch.int).detach().cpu().squeeze().numpy()
        pos = int((length-1) *(i+1) / num)
        color = np.array((pos%256, pos//256%256, pos//(1<<16)))
        gt_mask_show[label_val!=0] = color
        pred_mask_show[mask!=0] = color
    
    cv2.imwrite(filename+'_gt_color.jpg', gt_mask_show)
    cv2.imwrite(filename+'_color.jpg', pred_mask_show)


def record_iou(filename, ious, boundary_ious):
    if len(ious) == 0: return

    for i, (iou, biou) in enumerate(zip(ious, boundary_ious)):
        with open(filename+'_'+str(i)+'.txt','w') as f:
            f.write(str(round(iou.item()*100,2)))
  
def record_stability(filename, stabilitys, stabilityious):
    if len(stabilitys) == 0: return

    for i, iou in enumerate(stabilitys):
        with open(filename+'_'+str(i)+'_stability.txt','w') as f:
            f.write(str(round(iou.item()*100,2)))
        
        for j in range(args.stable_iter):
            with open(filename+'_'+str(i)+'_'+str(j)+'.txt','w') as f:
                f.write(str(round(stabilityious[i * args.stable_iter + j].item()*100,2)))
            
def show_points(coords, labels, ax, marker_size=500):
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
      
def show_box(box, ax, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=12))    

def compute_stability_refer_StableSAM(masks, labels_ori):
    
    N, H, W = labels_ori.shape
    groups = masks.shape[0] // N
    
    masks_binary =  (masks > 0).to(torch.float32) 
    masks_binary = masks_binary.view(groups, N, *masks_binary.shape[-2:])
    masks_union = torch.any(masks_binary, dim=0, keepdim=False).unsqueeze(1).to(torch.float32)*255.0
    
    stability, stability_list = 0, []
    
    for i in range(groups):
        #import pdb; pdb.set_trace()
        #print(masks_union.shape, labels_ori.shape,masks.shaoe)
        if args.eval_stability_with == 'union': x, y = compute_iou(masks[i*N:i*N+N,...], masks_union )
        if args.eval_stability_with == 'gt': 
            x, y = compute_iou(masks[i*N:i*N+N,...], labels_ori.unsqueeze(1))
                
        stability += x
        stability_list.extend(y)

    iou_list = copy.deepcopy(stability_list)
    stability_list = [sum(stability_list[i::N])/groups for i in range(N)]
    
    return stability / groups, stability_list, iou_list

def get_args_parser():
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    parser.add_argument("--restore-sam-model-keys", type=str, default=None,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument('--train-datasets', nargs='+')
    parser.add_argument('--valid-datasets', nargs='+')
    parser.add_argument('--load_prefix', default='')
    parser.add_argument('--data_augmentation', type=str2bool, default=True)
    parser.add_argument('--input_batch', type=int, default=-1)
    
    # SAM setting
    parser.add_argument("--sam-type", type=str, default="sam", 
                        help="The type of model to load, in ['sam' ,'sam2', 'sam2.1', 'efficient-sam']")
        
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--model-config", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--baseline', action='store_true')
    
    # Tuning Decoder setting
    parser.add_argument('--tuning_manner', default='output_tokens', choices=['output_tokens','decoder', 'full_weights', 'lora', 'adaptor'])
    parser.add_argument('--adaptor_config', type=str, default="")
    parser.add_argument('--two_stage', type=str2bool, default=True)
    
    # Base Learning Rate Setting & Epoch
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--max_epoch_num', default=10, type=int)
    parser.add_argument('--serial_prompt', type=str2bool, default=True)
    parser.add_argument('--serial_prompt_size', type=int, default=900)

    # loss
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    
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
    parser.add_argument('--train_img_num', default=-1, type=int)
    parser.add_argument('--eval_img_num', default=-1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--eval_prompt_type', default='boxes')
    parser.add_argument('--train_prompt_types', nargs='+')
    parser.add_argument('--train_point_num', nargs='+')
    parser.add_argument('--point_num', type=int, default=10)

    # Prompt Stability Setting    
    parser.add_argument('--eval_stability',  type= str2bool, default=False)
    parser.add_argument('--eval_stability_with',  type= str, default='gt')
    parser.add_argument('--stable_iter', default=10, type=int)
    parser.add_argument('--boxes_noise_scale', default=0.2, type=float)
    parser.add_argument('--points_noise_scale', default=0.2, type=float)

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
    parser.add_argument('--eval_record', type=str2bool, default=False)
    parser.add_argument('--visualize2', action='store_true')
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--eval_multimask_output',  type= str2bool, default=False)
    parser.add_argument('--mask_id', default=0, type=int)
    
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def main(train_datasets, valid_datasets, args):
    ### --- Step 0: Initialize ---
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
            
    setup_seed(args.seed + args.local_rank)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train", limit=args.train_img_num)
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = None,
                                                        batch_size = args.batch_size_train,
                                                        batch_size_prompt = args.batch_size_prompt,
                                                        batch_size_prompt_start = args.batch_size_prompt_start,
                                                        training = True,
                                                        numworkers=args.numworkers,
                                                        args=args)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid", limit=args.eval_img_num)
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          batch_size_prompt = args.batch_size_prompt,
                                                          batch_size_prompt_start = args.batch_size_prompt_start,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: Model for DistributedDataParallel---
    if args.sam_type == 'sam':
        net = MaskDecoder_Tuning(args.model_type, args.tuning_manner) 
        if args.compile: net = torch.compile(net)
        if torch.cuda.is_available(): net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        net_without_ddp = net.module
        
        if args.restore_model:
            print("restore model from:", args.restore_model)
            net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        parameters_grad, parameters_no_grad = 0, 0
        for n,p in net_without_ddp.named_parameters():
            if p.requires_grad: parameters_grad += 1
            else: parameters_no_grad += 1
        print("Second Stage Decoder parameters_grad Number:", parameters_grad)
        print("Second Stage Decoder parameters_no_grad Number:", parameters_no_grad)
    
    # To Do
    elif args.sam_type == 'sam2' or args.sam_type == 'sam2.1':
        net = MaskDecoder_Tuning('vit_b', args.tuning_manner) 
        if args.compile: net = torch.compile(net)
        if torch.cuda.is_available(): net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        net_without_ddp = net.module
        
        
    if args.sam_type == 'sam':
        sam_checkpoint_map = {
            'vit_b': os.path.join(args.load_prefix,'../pretrained/sam_vit_b_01ec64.pth'),
            'vit_l': '../pretrained/sam_vit_l_0b3195.pth',
            'vit_h': '../pretrained/sam_vit_h_4b8939.pth',
        }
    elif args.sam_type == 'sam2':
        sam_checkpoint_map = {
            'vit_t':  '../pretrained/sam2_hiera_tiny.pt',
        }
    elif args.sam_type == 'sam2.1':
        sam_checkpoint_map = {
            'vit_t':  '../pretrained/sam2.1_hiera_tiny.pt',
        }

    if args.sam_type == 'sam':
        if 'adaptor' in args.tuning_manner:
            with open(args.adaptor_config, 'r') as f:
                adaptor_config = yaml.load(f, Loader=yaml.FullLoader)
            sam = models.make(adaptor_config['model']).cuda()
            for name, para in sam.named_parameters():
                if "image_encoder" in name and "prompt_generator" not in name:
                    para.requires_grad_(False)
            sam.load_state_dict(torch.load(sam_checkpoint_map[args.model_type]), strict=False)
        else: sam = sam_model_registry[args.model_type](sam_checkpoint_map[args.model_type])
        
        if args.tuning_manner == 'lora': sam = LoRA_Sam(sam, r=4).cuda() 
    elif args.sam_type == 'sam2' or args.sam_type == 'sam2.1':
        sam = build_sam2(args.model_config, sam_checkpoint_map[args.model_type])
        predictor = SAM2ImagePredictor(sam)
        
        # ['output_tokens','decoder']
        if args.tuning_manner ==  'output_tokens':
            for n,p in sam.named_parameters():
                if 'mask_tokens' not in n and 'iou_token' not in n and 'obj_score_token' not in n:
                    p.requires_grad = False
                else :
                    print('SAM2:', n, 'need gradient', p.shape)
                    
    parameters_grad, parameters_no_grad = 0, 0
    for n,p in sam.named_parameters():
        if p.requires_grad: parameters_grad += 1
        else: parameters_no_grad += 1
    print("First Stage SAM parameters_grad Number:", parameters_grad)
    print("First Stage SAM parameters_no_grad Number:", parameters_no_grad)
        
    if args.compile: sam = torch.compile(sam)
    _ = sam.to(device=args.device)
    
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    if args.restore_sam_model:
        print("restore sam model from:", args.restore_sam_model)
        if args.tuning_manner in ['lora']:
            sam.module.load_lora_parameters(args.restore_sam_model)
        else:
            checkpoint = torch.load(args.restore_sam_model,map_location="cpu")
            if args.restore_sam_model_keys: checkpoint = checkpoint[args.restore_sam_model_keys]
            sam.module.load_state_dict(checkpoint)
    sam_withou_ddp = sam.module
    
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        
        if args.tuning_manner in ['lora','adaptor']: 
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, sam_withou_ddp.parameters()), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif (args.sam_type=='sam2' or args.sam_type=='sam2.1') or args.tuning_manner in ['full_weights']: optimizer = optim.Adam(sam_withou_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)        
        elif args.tuning_manner in ['output_tokens','decoder']: optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)        
            
        if not args.slow_start:
            lr_scheduler = StepLR(optimizer, args.lr_drop_epoch)
            lr_scheduler.last_epoch=args.start_epoch
        else:
            print("slow start & fast decay")
            lr_scheduler = LambdaLR(optimizer, lr_lambda)

        train(args, net, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        evaluate(args, net, sam, valid_dataloaders)


def train(args, net, sam,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    #evaluate(args, net, sam, valid_dataloaders)
    
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    net.train()
    _ = net.to(device=args.device)
        
    for epoch in range(epoch_start,epoch_num): 
        save_check_point(args, epoch, sam, net)
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
 
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
    
        for mini_batch_data in metric_logger.log_every(train_dataloaders,10):   
            data, adv_prompt_data, batch_adv_prompt_training = mini_batch_data[0], mini_batch_data[1], mini_batch_data[1]['adv_prompt_training']
            
            #print(batch_adv_prompt_training) #[True,False, False, False]
            
            images, labels, ori_im_path = data['image'].cuda(), data['label'].cuda(), data['ori_im_path']  # [K 3 1024 1024]   [K N 1024 1024]
            batch_adv_prompt_training = batch_adv_prompt_training.cuda()
            
            K, N, H, W =labels.shape
            labels = labels.reshape(K*N,H,W).cuda()  #[K*N 1024 1024]
                           
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()  #[K 1024 1024 3]
            
            # input prompt
            train_prompt_types = copy.copy(args.train_prompt_types)
        
            labels_box = misc.masks_to_boxes(labels) #[K*N 4]
            if 'gt_noisy_prompt' in data and data['gt_noisy_prompt'][0] == True: 
                labels_box = misc.box_noise(labels_box, data['boxes_noise_scale'][0].cpu().item()) #[K*N 4]
                #print("noisy boxes training", data['boxes_noise_scale'][0].cpu().item())
            
            try:
                labels_points = misc.masks_sample_points(labels, k=random.choice(args.train_point_num)) #[K*N 10 2]
            except:
                if 'points' in train_prompt_types: train_prompt_types.remove('points')
            
            labels_256 = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256) #[K*N 1 256 256] # 暂时不使用
            
            batched_input = []
            for bi in range(len(images_np)):
                dict_input = dict()
                input_image = torch.as_tensor(images_np[bi].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # [3 1024 1024]
                dict_input['image'] = input_image 
                train_prompt_type = random.choice(train_prompt_types)
                
                sparse_slice, dense_slice = slice(bi*N,(bi+1)*N), slice(bi*N,(bi+1)*N)
                if train_prompt_type == 'boxes':
                    dict_input['boxes'] = labels_box[sparse_slice,...]  #N*4
                    
                elif train_prompt_type == 'points':
                    point_coords = labels_points[sparse_slice,...] # N k 2
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[:-1], device=point_coords.device) #[N k]
                else:
                    raise NotImplementedError
                
                dict_input['original_size'] = images_np[0].shape[:2]
                batched_input.append(dict_input)
            
            # batch_adv_prompt_training:
            adv_boxes, adv_points, adv_prompt_images, adv_prompt_labels = adv_prompt_data['adv_boxes'].cuda().view(-1,4).contiguous(), adv_prompt_data['adv_points'].cuda().view(K*N,-1,2).contiguous(),\
                adv_prompt_data['image'].cuda(), adv_prompt_data['label'].cuda().reshape(K*N,H,W)

            # [K*N,4] [K*N 10 2]
            adv_prompt_images_np = adv_prompt_images.permute(0, 2, 3, 1).cpu().numpy()  #[K 1024 1024 3]
            
            for bi in range(len(adv_prompt_images_np)):
                dict_input = dict()
                input_image = torch.as_tensor(adv_prompt_images_np[bi].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # [3 1024 1024]
                dict_input['image'] = input_image 

                train_prompt_type = random.choice(train_prompt_types)
                sparse_slice, dense_slice = slice(bi*N,(bi+1)*N), slice(bi*N,(bi+1)*N)
                
                if train_prompt_type == 'boxes':
                    dict_input['boxes'] = adv_boxes[sparse_slice,...]  #N*4
                    
                elif train_prompt_type == 'points':
                    point_coords = adv_points[sparse_slice,...] # N 10 2
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[:-1], device=point_coords.device) #[N 10]
                else:
                    raise NotImplementedError

                dict_input['original_size'] = adv_prompt_images_np[0].shape[:2]
                batched_input.append(dict_input)
            
            
            if (args.sam_type=='sam2' or args.sam_type=='sam2.1') or args.tuning_manner in ['full_weights', 'lora', 'adaptor']:
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
                masks = torch.cat([batched_output[i_l]['low_res_logits'].cuda() for i_l in range(len(batched_output))], dim=0)
                #print('full_weights tuning')
            if args.tuning_manner in ['output_tokens','decoder']:
                with torch.no_grad():
                    batched_output, interm_embeddings = sam(batched_input, multimask_output=False)    
                    
            if args.two_stage:                
                batch_len = len(batched_output)
                encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
                image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
                sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
                dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

                masks, ious = net(
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                ) # [K*N, 1 256, 256] logits
                        
            loss_mask, loss_dice =loss_masks(masks[:K*N,:,:,:], labels.unsqueeze(1)/255.0, K*N) 
            loss_mask, loss_dice = loss_mask * args.alpha, loss_dice*args.alpha
            loss = loss_mask + loss_dice
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}
                
            # batch_adv_prompt_training loss:
            loss_mask_adv_prompt, loss_dice_adv_prompt =  loss_masks(masks[K*N:,:,:,:], adv_prompt_labels.unsqueeze(1)/255.0, K*N, weight=batch_adv_prompt_training.unsqueeze(1).expand(K,N).contiguous().view(K*N))
            loss_mask_adv_prompt, loss_dice_adv_prompt = loss_mask_adv_prompt*args.beta, loss_dice_adv_prompt*args.beta
            loss_adv_prompt = loss_mask_adv_prompt + loss_dice_adv_prompt
            loss_dict["loss_mask_adv_prompt"],loss_dict["loss_dice_adv_prompt"] = loss_mask_adv_prompt, loss_dice_adv_prompt
            loss += loss_adv_prompt
            
            #import pdb; pdb.set_trace()
            #print(loss.shape)
            #print(loss_dict, torch.sum(adv_boxes), torch.sum(labels_box))
            
            # reduce losses over all GPUs for logging purposes
            # print(loss)
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

            
            ## Debug for adv prompt training
            # if train_prompt_type =='boxes':
            #     save_dir = os.path.join(args.output, train_prompt_type, random.choice(train_datasets)['name'])
            # if train_prompt_type =='points':
            #     save_dir = os.path.join(args.output, train_prompt_type+'_'+str(args.point_num), random.choice(train_datasets)['name'])
            
            # Path(save_dir).mkdir(parents=True,exist_ok=True)
            
            # for batch_i in range(images.shape[0]):
                
            #     base = ori_im_path[batch_i].split('/')[-1]
            #     index = base[::-1].index('.')
            #     base = base[:-index-1]
            #     save_base = os.path.join(save_dir, str(base))
            #     #print(save_base)
                
            #     masks_vis = (F.interpolate(masks[K*N+batch_i*N:K*N+batch_i*N+N,:,:,:].detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu() # [N,1,1024,1024]
            #     adv_prompt_image_np = adv_prompt_images_np[batch_i].astype(dtype=np.uint8)  #[1024 1024 3]
            #     cv2.imwrite(save_base+'_im.jpg',adv_prompt_image_np[:,:,::-1])
                
            #     #print(masks_vis.shape)
            #     iou, boundary_iou = [0]*N, [0]*N 
                
            #     batch_slice = slice(batch_i*N,(batch_i+1)*N) 
                
            #     if train_prompt_type =='boxes':
            #         show_anns(adv_prompt_labels[batch_slice].cpu(), masks_vis, None, None, adv_boxes[batch_slice].cpu(), save_base , adv_prompt_image_np, iou, boundary_iou)
            #     elif train_prompt_type =='points':
            #         show_anns(adv_prompt_labels[batch_slice].cpu(), masks_vis, adv_points[batch_slice].cpu(), torch.ones(adv_points[batch_slice].shape[:2]).cpu(), None, save_base, adv_prompt_image_np, iou, boundary_iou)
            #     else:
            #         raise NotImplementedError

            ## Debug for normal training
            # if train_prompt_type =='boxes':
            #     save_dir = os.path.join(args.output, train_prompt_type, random.choice(train_datasets)['name'])
            # if train_prompt_type =='points':
            #     save_dir = os.path.join(args.output, train_prompt_type+'_'+str(args.point_num), random.choice(train_datasets)['name'])
            
            # Path(save_dir).mkdir(parents=True,exist_ok=True)
            
            # for batch_i in range(images.shape[0]):
                
            #     base = ori_im_path[batch_i].split('/')[-1]
            #     index = base[::-1].index('.')
            #     base = base[:-index-1]
            #     save_base = os.path.join(save_dir, str(base)+'_normal')
            #     #print(save_base)
                
            #     masks_vis = (F.interpolate(masks[batch_i*N:batch_i*N+N,:,:,:].detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu() # [N,1,1024,1024]
            #     image_np = images_np[batch_i].astype(dtype=np.uint8)  #[1024 1024 3]
            #     cv2.imwrite(save_base+'_im.jpg',image_np[:,:,::-1])
                
            #     #print(masks_vis.shape)
            #     iou, boundary_iou = [0]*N, [0]*N 
                
            #     batch_slice = slice(batch_i*N,(batch_i+1)*N) 
                
            #     if train_prompt_type =='boxes':
            #         show_anns(labels[batch_slice].cpu(), masks_vis, None, None, labels_box[batch_slice].cpu(), save_base , image_np, iou, boundary_iou)
            #     elif train_prompt_type =='points':
            #         show_anns(labels[batch_slice].cpu(), masks_vis, labels_points[batch_slice].cpu(), torch.ones(labels_points[batch_slice].shape[:2]).cpu(), None, save_base, image_np, iou, boundary_iou)
            #     else:
            #         raise NotImplementedError
            
    
        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(f"Epoch {str(epoch)}: "+str(train_stats)[1:-1]+'\n')
        
        lr_scheduler.step()
        dist.barrier()
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0: save_check_point(args, epoch, sam, net)

            
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and tune_decoder
    if args.sam_type!='sam2' and args.tuning_manner in ['output_tokens','decoder'] and  misc.is_main_process():        
        sam_checkpoint_map = {
            'vit_b': '../pretrained/sam_vit_b_01ec64.pth',
            'vit_l': '../pretrained/sam_vit_b_01ec64.pth',
            'vit_h': '../pretrained/sam_vit_b_01ec64.pth',
        }
        sam_ckpt = torch.load(sam_checkpoint_map[args.model_type]) 

        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            if 'mask_token' in key or 'iou_token' in key:
                sam_key = 'mask_decoder.'+key
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/asam_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)


def save_check_point(args, epoch, sam, net):
    model_name = "/epoch_"+str(epoch)+".pth"
    print('come here save at', args.output + model_name)
    if (args.sam_type=='sam2' or args.sam_type=='sam2.1')or args.tuning_manner in ['full_weights', 'adaptor']:
        misc.save_on_master(sam.module.state_dict(), args.output + model_name)
    elif args.tuning_manner in ['lora'] and misc.is_main_process():
        sam.module.save_lora_parameters(args.output + model_name)            
    elif args.tuning_manner in ['output_tokens','decoder']:
        misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
@torch.no_grad()
def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    iou_list = []
    #print(len(preds))
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

def bad_examples_info(bad_examples, metric_logger, k):
    bad_examples[0] += 1
    loss_dict = {"val_iou_"+str(k): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(k): torch.tensor(0.5).cuda()}
    if args.eval_stability: loss_dict
    
    loss_dict_reduced = misc.reduce_dict(loss_dict)
    metric_logger.update(**loss_dict_reduced)

@torch.no_grad()
def evaluate_one_sample(args, net, sam, bad_examples, dataset_id, metric_logger,  iou_head_prediction_sum, all_mask_nums, max_len, data_val):
    imidx_val, images_val, labels_val, shapes_val, labels_ori, ori_im_path = data_val['imidx'], data_val['image'].cuda(), data_val['label'].cuda(), data_val['shape'], data_val['ori_label'],data_val['ori_im_path']
    
    # print(labels_val.shape)
    # max_len[0] = max(max_len[0], labels_val.shape[1])
    
    # labels_val = labels_val[0:1,0:1,:,:].expand(-1, max_len[0], -1, -1)
    # labels_ori = labels_ori[0:1,0:1,:,:].expand(-1, max_len[0], -1, -1)
    
    K,N,H,W = labels_val.shape
    K,N,h,w = labels_ori.shape
    
    if N == 0: bad_examples_info(bad_examples,metric_logger,k); print_current_line(sys._getframe()); return
        
    labels_val = labels_val.reshape(K*N,H,W).cuda() #K*N 1024 1024 
    labels_ori = labels_ori.reshape(K*N,h,w).cuda()
                
    images_val_np = images_val.permute(0, 2, 3, 1).cpu().numpy() # K 3 1024 1024 -> k 1024 1024 3
    
    if args.eval_prompt_type=='boxes':
        labels_box = misc.masks_to_boxes(labels_val) #K*N 4    
    if args.eval_prompt_type=='points':  
        try: labels_points = misc.masks_sample_points(labels_val,k=args.point_num) #[K*N 10 2]
        except: bad_examples_info(bad_examples,metric_logger,k); print_current_line(sys._getframe()); return
    
    batched_input = []
    dict_input = dict()
    
    input_image = torch.as_tensor(images_val_np[0].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # 3 1024 1024
    dict_input['image'] = input_image
    if args.eval_prompt_type == 'boxes':
        dict_input['boxes'] = labels_box #N 4
    elif args.eval_prompt_type == 'points':
        point_coords = labels_points #[N 10 2]
        dict_input['point_coords'] = point_coords
        dict_input['point_labels'] = torch.ones(point_coords.size()[:2], device=point_coords.device)
    else:
        raise NotImplementedError
    
    if args.eval_stability and args.eval_prompt_type == 'boxes':
        dict_input['boxes'] = torch.cat([dict_input['boxes'],torch.cat([misc.box_noise(dict_input['boxes'], args.boxes_noise_scale) for i in range(args.stable_iter)],dim=0)])
    
    ## To do
    if args.eval_stability and args.eval_prompt_type == 'points':
        for i in range(args.stable_iter):
            dict_input['point_coords'] = torch.cat([dict_input['point_coords'],misc.masks_sample_points(labels_val,k=args.point_num)])
        dict_input['point_labels'] = torch.ones(dict_input['point_coords'].size()[:2], device=point_coords.device)
        #import pdb; pdb.set_trace()
        
    dict_input['original_size'] = images_val_np[0].shape[:2]
    batched_input.append(dict_input)

    
    if args.serial_prompt:
        new_batched_input = []
        for i in range((labels_ori.shape[0] * (args.stable_iter if args.eval_stability else 0) + labels_ori.shape[0] + args.serial_prompt_size -1 )// args.serial_prompt_size):
            start, end = i*args.serial_prompt_size, min(i*args.serial_prompt_size+args.serial_prompt_size, labels_ori.shape[0] * args.stable_iter + labels_ori.shape[0])
            if end - start < 1: return
            
            serial_slice = slice(start, end)
            new_dict = {}
            new_dict['image'] = batched_input[0]['image']
            if args.eval_prompt_type == 'boxes': new_dict['boxes'] = batched_input[0]['boxes'][serial_slice]
            if args.eval_prompt_type == 'points': 
                new_dict['point_coords'] = batched_input[0]['point_coords'][serial_slice]
                new_dict['point_labels'] = batched_input[0]['point_labels'][serial_slice]
            new_dict['original_size'] = images_val_np[0].shape[:2]
            
            new_batched_input.append(new_dict)
            
        
        batched_input = new_batched_input
    
    with torch.autocast(device_type="cuda"):
        batched_output, interm_embeddings = sam(batched_input, multimask_output=args.eval_multimask_output, multiplexing_mode=True)
        
    batch_len = len(batched_output)
    masks = torch.cat([batched_output[i_l]['low_res_logits'].cuda() for i_l in range(batch_len)], dim=0)
    ious = torch.cat([batched_output[i_l]['iou_predictions'] for i_l in range(batch_len)], dim=0)

    #import pdb; pdb.set_trace()
    if args.baseline or args.two_stage == False:
        masks = masks.to(torch.float32) if not args.eval_multimask_output else masks[:,args.mask_id:args.mask_id+1,:,:].to(torch.float32)
        ious = ious.to(torch.float32) if not args.eval_multimask_output else ious[:,args.mask_id:args.mask_id+1].to(torch.float32)
        iou_head_prediction_sum[0] += torch.sum(ious).item()
        all_mask_nums[0] += torch.numel(ious)
        #print('single stage testing')
    else:
        # print(args.eval_multimask_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
        masks, ious = net(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.eval_multimask_output,
        ) #[N,:,1024,1024]
        
        #import pdb; pdb.set_trace()
        if args.eval_multimask_output: 
            ious = ious[:,args.mask_id:args.mask_id+1]
            masks = masks[:,args.mask_id:args.mask_id+1,:,:]
        
        masks = F.interpolate(masks, scale_factor=4, mode='bilinear', align_corners=False)
        iou_head_prediction_sum[0] += torch.sum(ious).item()
        all_mask_nums[0] += torch.numel(ious)
        
    # print(masks.shape, ious.shape)
    # raise NameError
    
    # iou,iou_list = compute_iou(masks[:labels_ori.shape[0]],labels_ori.unsqueeze(1))
    # boundary_iou,boundary_iou_list = compute_boundary_iou(masks[:labels_ori.shape[0]],labels_ori.unsqueeze(1))               
    
    try:
        iou,iou_list = compute_iou(masks[:labels_ori.shape[0]],labels_ori.unsqueeze(1))
        boundary_iou,boundary_iou_list = compute_boundary_iou(masks[:labels_ori.shape[0]],labels_ori.unsqueeze(1))               
        if args.eval_stability: stability, stability_list, stability_iou_list = compute_stability_refer_StableSAM(masks[labels_ori.shape[0]:], labels_ori)
    except: bad_examples_info(bad_examples,metric_logger,k); print_current_line(sys._getframe()); return
    
    #print(iou)
    
    if torch.isnan(iou).any() or torch.isnan(boundary_iou).any(): bad_examples_info(bad_examples,metric_logger,k); print_current_line(sys._getframe()); return

    if args.eval_prompt_type =='boxes':
        save_dir = os.path.join(args.output, args.eval_prompt_type, valid_datasets[dataset_id]['name'])
    if args.eval_prompt_type =='points':
        save_dir = os.path.join(args.output, args.eval_prompt_type+'_'+str(args.point_num), valid_datasets[dataset_id]['name'])
    
    Path(save_dir).mkdir(parents=True,exist_ok=True)
    
    base = ori_im_path[0].split('/')[-1]
    index = base[::-1].index('.')
    base = base[:-index-1]
    save_base = os.path.join(save_dir, str(base))
    Path(save_base).mkdir(parents=True,exist_ok=True)
    
    if args.eval_record: record_iou(save_base, iou_list, boundary_iou_list)
    if args.eval_stability and args.eval_record: record_stability(save_base, stability_list, stability_iou_list)
    
    #print(os.path.join(args.output, 'GT', valid_datasets[dataset_id]['name'],str(base))+'.png')
    
    # save GT
    #Path(os.path.join(args.output, 'GT', valid_datasets[dataset_id]['name'])).mkdir(parents=True,exist_ok=True)
    #cv2.imwrite(os.path.join(args.output, 'GT', valid_datasets[dataset_id]['name'],str(base))+'.png', labels_ori[0,...].detach().cpu().numpy())

    if args.visualize:
        # standard visualize
        masks_vis = (F.interpolate(masks[:labels_val.shape[0]].detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
        image_val_np = images_val_np[0].astype(dtype=np.uint8) #[1024,1024,3]
        cv2.imwrite(save_base+'_im.jpg',image_val_np[:,:,::-1])
        if args.eval_prompt_type=='boxes':
            show_anns(labels_val.cpu(), masks_vis, None, None, labels_box.cpu(), save_base , image_val_np, iou_list, boundary_iou_list, boxes_color='gold')
        elif args.eval_prompt_type=='points':
            show_anns(labels_val.cpu(), masks_vis, labels_points.cpu(), torch.ones(labels_points.shape[:2]).cpu(), None, save_base , image_val_np, iou_list, boundary_iou_list)

        # adv prompt visualize
        for i in range(args.stable_iter):
            length = labels_val.shape[0]
            masks_vis = (F.interpolate(masks[length*(i+1): length*(i+2)].detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
            image_val_np = images_val_np[0].astype(dtype=np.uint8) #[1024,1024,3]
            
            if args.eval_prompt_type=='boxes':
                show_anns(labels_val.cpu(), masks_vis, None, None,  dict_input['boxes'][length*(i+1): length*(i+2)].cpu(), save_base, image_val_np, iou_list, boundary_iou_list, save_gt=False, suffix=str(i), boxes_color='red')
            elif args.eval_prompt_type=='points':
                show_anns(labels_val.cpu(), masks_vis, dict_input['point_coords'][length*(i+1): length*(i+2)].cpu(), torch.ones(dict_input['point_coords'][length*(i+1): length*(i+2)].shape[:2]).cpu(), None, save_base, image_val_np, iou_list, boundary_iou_list, save_gt=False, suffix=str(i))
            
    ## bug to fix
    if args.visualize2:
        masks_vis = (F.interpolate(masks.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
        imgs_ii = imgs[0].astype(dtype=np.uint8)
        
        if args.prompt_type=='box':
            show_anns2(labels_val.cpu(), masks_vis, None, labels_box.cpu(), None, save_base , imgs_ii, iou_list, boundary_iou_list)
        elif args.prompt_type=='point':
            show_anns2(labels_val.cpu(), masks_vis, labels_points.cpu(), None, torch.ones(labels_points.shape[:2]).cpu(), save_base , imgs_ii, iou_list, boundary_iou_list)
    
    loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): iou.cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): boundary_iou.cuda()}
    if args.eval_stability: loss_dict['val_stability_'+str(valid_datasets[dataset_id]['name'])] = stability.cuda()
    
    loss_dict_reduced = misc.reduce_dict(loss_dict)
    metric_logger.update(**loss_dict_reduced)
    
    # 等待所有进程到达这里
    dist.barrier()
            
@torch.no_grad()
def evaluate(args, net, sam, valid_dataloaders):
    net.eval()
    print("Validating...")
    test_stats = {}
     
    max_len = [0] 
    for dataset_id, k in enumerate(range(len(valid_dataloaders))):
        bad_examples = [0]
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        iou_head_prediction_sum, all_mask_nums = [0], [0]
        print('valid_dataloader len:', len(valid_dataloader), valid_datasets[dataset_id]['name'])
        
        
        for data_val in metric_logger.log_every(valid_dataloader,10):
            evaluate_one_sample(args, net, sam, bad_examples, dataset_id, metric_logger,  iou_head_prediction_sum, all_mask_nums, max_len, data_val)
            try: evaluate_one_sample(args, net, sam, bad_examples, dataset_id, metric_logger,  iou_head_prediction_sum, all_mask_nums, max_len, data_val)
            except: bad_examples_info(bad_examples,metric_logger,k), print_current_line(sys._getframe()); 
            
        print(max_len[0])
        #import pdb; pdb.set_trace()
        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        print((str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n'))
        
        if args.eval_multimask_output:                
            print(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
        else:
            print(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                    
        text_log = {k: round(meter.global_avg*100,2) for k, meter in metric_logger.meters.items() if meter.count > 0}
        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(str(valid_datasets[dataset_id]['name'])+' '+ str(text_log)[1:-1].replace("'","")+'\n')
                f.write(str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n') 
                if args.eval_multimask_output:                
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                else:
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n')
                    
        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        print((str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n'))
        
        if args.eval_multimask_output:                
            print(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
        else:
            print(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                    
        text_log = {k: round(meter.global_avg*100,2) for k, meter in metric_logger.meters.items() if meter.count > 0}
        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(str(valid_datasets[dataset_id]['name'])+' '+ str(text_log)[1:-1].replace("'","")+'\n')
                f.write(str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n') 
                if args.eval_multimask_output:                
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                else:
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        print((str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n'))
        
        if args.eval_multimask_output:                
            print(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
        else:
            print(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                    
        text_log = {k: round(meter.global_avg*100,2) for k, meter in metric_logger.meters.items() if meter.count > 0}
        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(str(valid_datasets[dataset_id]['name'])+' '+ str(text_log)[1:-1].replace("'","")+'\n')
                f.write(str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples[0]) +'\n') 
                if args.eval_multimask_output:                
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head for mask' + str(args.mask_id) + ': '+ str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 
                else:
                    f.write(str(valid_datasets[dataset_id]['name'])+' iou_head: ' + str(iou_head_prediction_sum[0]/all_mask_nums[0]) +'\n') 


    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------
        
    ## Train dataset
    dataset_lecun = {"name": "lecun_sam_subset",
        "im_dir": "../lecun_photo",
        "gt_dir": "../lecun_photo",
        "im_ext": ".jpg",
        "gt_ext": ""
    }
    
    dataset_lecun_adv = {"name": "lecun_sam_subset_adv",
        "im_dir": "../output/lecun-Grad/Attacker-7.5-50-Definder-sam-vit_b-140-point-10-Loss-100.0-100.0-100.0-1.0-2-Perturbation-0.2-10-0.02-0.5/adv",
        "gt_dir": "../lecun_photo",
        "im_ext": ".png",
        "gt_ext": ""
    }
    
    dataset_sa000000 = {"name": "sam_subset_ori",
        "im_dir": "../sam-1b/sa_000000",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-0-AttackObject-boxes-Definder-sam-vit_b-4-box-10-Loss-100.0-100.0-0.5-2-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-0-AttackObject-points-Definder-sam-vit_b-4-points-10-Loss-100.0-100.0-0.5-2-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
    }
    
    dataset_sa000000_adv_imgs_prompt = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad_xx/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-1.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad_xx/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-1.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad_xx/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-1.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
    }

    dataset_sa000138_dci = {"name": "sam1b_sa000138_dci",
        "im_dir": "../output/sa_000138-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.4-10-0.04-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".png",
        "gt_ext": ""
    }

    dataset_sa000138_controlnet_early= {"name": "sam1b_sa000138_controlnet",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/.._ckpt_control_v11p_sd15_mask_sa000001.pth_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }

    dataset_sa000138_controlnet= {"name": "sam1b_sa000138_controlnet",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/.._ckpt_control_v11p_sd15_mask_sa000002.pth_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }

    dataset_sa000138_controlnet_plus_no_reward_1 = {"name": "sam1b_sa000138_controlnet_plus",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/work_dirs_reward_model_SAM-1B_reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_no_reward_checkpoint-1_controlnet_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }

    dataset_sa000138_controlnet_plus_10 = {"name": "sam1b_sa000138_controlnet_plus",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/work_dirs_reward_model_SAM-1B_reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_restart_checkpoint-10_controlnet_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }
    
    dataset_sa000138_controlnet_plus_500 = {"name": "sam1b_sa000138_controlnet_plus",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/work_dirs_reward_model_SAM-1B_reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_checkpoint-500_controlnet_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }
    
    dataset_sa000138_controlnet_plus_4500 = {"name": "sam1b_sa000138_controlnet_plus",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/work_dirs_reward_model_SAM-1B_reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_checkpoint-4500_controlnet_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }

    dataset_sa000138_controlnet_plus_7000 = {"name": "sam1b_sa000138_controlnet_plus",
        "im_dir": "../ControlNet_Plus_Plus/work_dirs/eval_dirs/sa000000/validation/work_dirs_reward_model_SAM-1B_reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_checkpoint-7000_controlnet_7.5-20/images/group_1",
        "gt_dir": "../sam-1b/sa_000138",
        "im_ext": ".jpg",
        "gt_ext": ""
    }
    
    dataset_sa000000efficient = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam_efficient-vit_t-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""
    }
    
    dataset_sa000000adv = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-100-1-2-10-Clip-0.2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""
    }

    dataset_sa000000_direct_inversion_adv_img_adv_prompt = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': False,
        'adv_prompt': True,
    }
    
    dataset_sa000000_direct_inversion_adv_augmentation_img_adv_prompt = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': True,
    }

    dataset_sa000000_direct_inversion_adv_augmentation_img_adv_prompt_sam2 = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad-SAM2/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam2-vit_t-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-100.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad-SAM2/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam2-vit_t-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-100.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': True,
    }


    dataset_sa000000_direct_inversion_adv_augmentation_img_gt_prompt_sam2 = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad-SAM2/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam2-vit_t-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-100.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad-SAM2/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam2-vit_t-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-100.0-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': False,
    }
    
    dataset_sa000000_direct_inversion_adv_augmentation_img_adv_prompt_controlnet_plus_plus = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad-ControlNet++/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad-ControlNet++/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad-ControlNet++/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': True,
    }
    
    dataset_sa000000_direct_inversion_adv_augmentation_img_gt_prompt = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': False,
    }


    dataset_sa000000_direct_inversion_adv_augmentation_img_gt_noisy_prompt = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'augmentation': True,
        'adv_prompt': False,
        'gt_noisy_prompt': True,
        'boxes_noise_scale': 0.2,
    }
    
    
    dataset_sa000000_ori_img_adv_prompt = {"name": "sam_subset",
        "im_dir": "../sam-1b/sa_000000",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'aumentation': False,
        'adv_prompt': True,
    }

    dataset_sa000000_ori_augmentation_img_gt_prompt = {"name": "sam_subset",
        "im_dir": "../sam-1b/sa_000000",
        "gt_dir": "../sam-1b/sa_000000",
        "adv_boxes_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',
        "adv_points_dir": '../output/sa_000000-Grad/Attacker-7.5-50-AttackObject-image_points_boxes-Definder-sam-vit_b-4-points_boxes-10-Loss-100.0-100.0-0.5-2-embedding_sup-0.5-Perturbation_Img-0.2-10-0.01-0.5-Perturbation_Boxes-20.0-10-4.0-0.5-0.1-Perturbation_Points-20.0-10-4.0-0.5-0.1/adv',        
        "im_ext": ".jpg",
        "gt_ext": "",
        "adv_boxes_ext": '_boxes.txt',
        "adv_points_ext": '_points.txt',
        'aumentation': True,
        'adv_prompt': False,
    }
    

    dataset_sa000000adv_dice = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""
    }
    
    dataset_sa000001adv_dice = {"name": "sam_subset",
        "im_dir": "../output/sa_000001-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000001",
        "im_ext": ".png",
        "gt_ext": ""
    }
    
    dataset_sa000000_Inversion = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Inversion/inv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""
    }
            
    dataset_DatasetDM = {"name": "DatasetDM",
        "im_dir": "../DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/Image",
        "gt_dir": "../DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/label",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    
    dataset_sa000000pgd = {"name": "sam_subset",
        "im_dir": "work_dirs/PGD",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ".json"
    }
    
    ## valid set
    dataset_hrsod_val = {"name": "HRSOD-TE",
        "im_dir": "data/HRSOD-TE/imgs",
        "gt_dir": "data/HRSOD-TE/gts",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ade20k_val = {"name": "ADE20K_2016_07_26",
        "im_dir": "data/ADE20K_2016_07_26/images/validation",
        "gt_dir": "data/ADE20K_2016_07_26/images/validation",
        "im_ext": ".jpg",
        "gt_ext": "_seg.png"
    }
    
    dataset_cityscapes_val = {"name": "cityscaps_val",
        "im_dir": "data/cityscapes/leftImg8bit/val",
        "gt_dir": "data/cityscapes/gtFine/val",
        "im_ext": "_leftImg8bit.png",
        "gt_ext": "_gtFine_instanceIds.png"
    }
    
    dataset_voc2012_val = {"name": "voc2012_val",
        "im_dir": "data/VOC2012/JPEGImages_val",
        "gt_dir": "data/VOC2012/SegmentationObject",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_coco2017_val = {"name": "coco2017_val",
        "im_dir": "data/COCO2017-val/val2017",
        "annotation_file": "data/COCO2017-val/instances_val2017.json",
        "im_ext": ".jpg"
    }
    
    dataset_camo = {"name": "camo",
        "im_dir": "data/CAMO/imgs",
        "gt_dir": "data/CAMO/gts",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ishape_antenna = {"name": "ishape***",
        "im_dir": "data/ishape/antenna/val/image",
        "gt_dir": "data/ishape/antenna/val/instance_map",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "data/Plant_Phenotyping",
        "gt_dir": "data/Plant_Phenotyping",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
    }
    
    dataset_pascal_part58 = {"name": "Pascal_Part58",
        "im_dir": "data/Pascal-Part-201/Img_val",
        "gt_dir": "data/Pascal-Part-201/parts58",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_pascal_part201 = {"name": "Pascal_Part201",
        "im_dir": "data/Pascal-Part-201/Img_val",
        "gt_dir": "data/Pascal-Part-201/parts201",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_pascal_part108 = {"name": "Pascal_Part108",
        "im_dir": "data/Pascal-Part-201/Img_val",
        "gt_dir": "data/Pascal-Part-201/parts108",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ImagenetPart = {"name": "ImagenetPart",
        "im_dir": "data/PartImageNet/images/test",
        "gt_dir": "data/PartImageNet/annotations/test",
        "im_ext": ".JPEG",
        "gt_ext": ".png"
    }
    
    dataset_TimberSeg = {"name": "timberseg_coco",
        "im_dir": "data/TimberSeg/prescaled/",
        "annotation_file": "data/TimberSeg/prescaled/coco_annotation_rotated.json",
        "im_ext": ".png",
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "data/Plant_Phenotyping",
        "gt_dir": "data/Plant_Phenotyping",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
    }
     
    dataset_streets = {"name": "streets",
        "im_dir": "data/Streets/images",
        "gt_dir": "data/Streets/labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_paco_lvis = {"name": "PACO_LVIS_coco",
        "im_dir": "data/PACO/",
        "annotation_file": "data/PACO/paco_lvis_v1_val.json",
        "im_ext": ".jpg",
    }
    
    dataset_big_val = {"name": "big",
        "im_dir": "data/BIG/val",
        "gt_dir": "data/BIG/val",
        "im_ext": "_im.jpg",
        "gt_ext": "_gt.png"
    }
    
    dataset_ndis_train = {"name": "ndis_park_coco",
        "im_dir": "data/ndis_park/train/imgs",
        "annotation_file": "data/ndis_park/train/train_coco_annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_test = {"name": "Plittersdorf_coco",
        "im_dir": "data/Plittersdorf/images",
        "im_dir": "data/Plittersdorf/images",
        "annotation_file": "data/Plittersdorf/test.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_val = {"name": "Plittersdorf_coco",
        "im_dir": "data/Plittersdorf/images",
        "annotation_file": "data/Plittersdorf/val.json",
        "im_ext": ".jpg",
    }
        
    dataset_egohos = {"name": "egohos",
        "im_dir": "data/egohos/val/image",
        "gt_dir": "data/egohos/val/label",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_LVIS = {"name": "LVIS***",
        "im_dir": "data/LVIS/val2017",
        "annotation_file": "data/LVIS/annotations/lvis_v1_val.json",
        "im_ext": ".jpg",
    }
    
    dataset_BBC038v1 = {"name": "BBC038v1",
        "im_dir": "data/BBC038V1-Train",
        "annotation_file": "data/BBC038V1-Train",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS1 = {"name": "DOORS1***",
        "im_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/img",
        "gt_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS2 = {"name": "DOORS2***",
        "im_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/img",
        "gt_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_NDD20_ABOVE = {"name": "NDD20_Above",
        "im_dir": "data/NDD20/ABOVE",
        "gt_dir": "data/NDD20/ABOVE_LABELS",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_NDD20_BELOW = {"name": "NDD20_Below",
        "im_dir": "data/NDD20/BELOW",
        "gt_dir": "data/NDD20/BELOW_LABELS",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }   
        
    dataset_PIDRAY = {"name": "pid_coco***",
        "im_dir": "data/pidray/hard",
        "annotation_file": "data/pidray/annotations/xray_test_hard.json",
        "im_ext": ".jpg",
    }
    
    dataset_TrashCan_val = {"name": "TrashCAN_coco",
        "im_dir": "data/TrashCan/instance_version/val",
        "annotation_file": "data/TrashCan/instance_version/instances_val_trashcan.json",
        "im_ext": ".jpg",
    }
    
    dataset_ZeroWaste = {"name": "ZeroWaste",
        "im_dir": "data/ZeroWaste/val/data",
        "gt_dir": "data/ZeroWaste/val/sem_seg",
        "im_ext": ".PNG",
        "gt_ext": ".PNG"
    }
    
    dataset_DRAM_test = {"name": "DRAM",
        "im_dir": "data/DRAM",
        "gt_dir": "data/DRAM",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ovis_train = {"name": "ovis",
        "im_dir": "data/OVIS/train_img",
        "gt_dir": "data/OVIS/train_labels",
        "im_ext": ".jpg",
        "gt_ext": ""
    }
    
    dataset_ibd_val = {"name": "ibd",
        "im_dir": "data/IBD/val",
        "gt_dir": "data/IBD/val_labels",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_visor_val = {"name": "visor_gtea",
        "im_dir": "data/VISOR/GroundTruth-SparseAnnotations/rgb_frames/val",
        "gt_dir": "data/VISOR/GroundTruth-SparseAnnotations/annotations/val",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_woodscape = {"name": "woodscape",
        "im_dir": "data/WoodScape/rgb_images",
        "gt_dir": "data/WoodScape/instance_annotations",
        "im_ext": ".png",
        "gt_ext": ".json"
    }
    
    dataset_gtea_train = {"name": "gtea***",
        "im_dir": "data/GTEA_hand2k/GTEA_GAZE_PLUS/Images",
        "gt_dir": "data/GTEA_hand2k/GTEA_GAZE_PLUS/Masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    # medical dataset
    dataset_Kvasir_SEG = {"name": "Kvasir_SEG",
        "im_dir": "data/Kvasir-SEG/images",
        "gt_dir": "data/Kvasir-SEG/masks",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    
    dataset_Kvasir_sessile = {"name": "Kvasir_sessile",
        "im_dir": "data/Kvasir-sessile/images",
        "gt_dir": "data/Kvasir-sessile/masks",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    
    dataset_CVC_ClinicDB = {"name": "CVC_ClinicDB",
        "im_dir": "data/CVC-ClinicDB/Original",
        "gt_dir": "data/CVC-ClinicDB/Ground Truth",
        "im_ext": ".tif",
        "gt_ext": ".tif"
    }
    
    args = get_args_parser()
    
    if not args.eval:
        args.output = os.path.join('work_dirs', args.output_prefix+'-')
        for train_dataset in args.train_datasets:
            args.output += train_dataset.replace('dataset_','')
            args.output += '-'
        args.output += args.model_type + '-' + str(args.train_img_num) 
    
    elif args.baseline: 
        args.output = os.path.join('work_dirs', args.output_prefix+'-'+args.model_type)
    elif args.restore_model:
        args.output = os.path.join(*args.restore_model.split('/')[:-1])
    elif args.restore_sam_model:
        args.output = os.path.join(*args.restore_sam_model.split('/')[:-1])
    
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
