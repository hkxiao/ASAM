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

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, batched_input):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, batched_input):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, batched_input):
        """
        """
        # diff = x - self.orig_input
        # diff = torch.clamp(diff, -self.eps, self.eps)
        # return torch.clamp(diff + self.orig_input, 0, 1)
    
        for input in batched_input: 
            diff = input['image'] - input['orig_image']
            diff = torch.clamp(diff, -self.eps, self.eps)
            #print(torch.max(diff))
            input['image'] = torch.clamp(input['orig_image'] + diff, 0, 255)
        return batched_input
    def step(self, batched_input):
        """
        """
        # step = torch.sign(g) * self.step_size
        # return x + step
    
        for input in batched_input:
            input['image'] = input['image'] + torch.sign(input['grad']) * self.step_size
        return batched_input

    def random_perturb(self, batched_input):
        """
        """
        for input in batched_input:
            l = len(input['image'].shape) - 1
            rp = torch.randn_like(input['image'])
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))            
            input['image'] =  torch.clamp(input['image'] + self.eps * rp / (rp_norm + 1e-10), 0, 255)
        return batched_input

class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, batched_input):
        """
        """
        for input in batched_input: 
            diff = input['image'] - input['orig_image']
            diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
            #print(torch.max(diff))
            input['image'] = torch.clamp(input['orig_image'] + diff, 0, 255)
            
        return batched_input

    def step(self, batched_input):
        """
        """
        for input in batched_input:
            l = len(input['image'].shape) - 1
            g_norm = torch.norm(input['grad'].view(input['grad'].shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_g = input['grad'] / (g_norm + 1e-10)
            input['image'] = input['image'] + scaled_g * self.step_size
        return batched_input
    
    def random_perturb(self, batched_input):
        """
        """
        for input in batched_input:
            l = len(input['image'].shape) - 1
            rp = torch.randn_like(input['image'])
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))            
            input['image'] =  torch.clamp(input['image'] + self.eps * rp / (rp_norm + 1e-10), 0, 255)
        return batched_input
    
def replace_best(loss, bloss, batched_input, m):
    if bloss is None:
        for input in batched_input:
            input['best_image'] = input['image'].clone().detach()
        bloss = loss.clone().detach()
    else:
        if args.only_attack: 
            replace = (m * bloss.mean() < m * loss.mean()).unsqueeze(0)
        else:
            replace = torch.tensor([True]*args.batch_size_train)
            for i in range(args.batch_size_train):
                replace[i] = (m * bloss[i*args.batch_size_prompt:(i+1)*i*args.batch_size_prompt,...].mean() < \
                    m * loss[i*args.batch_size_prompt:(i+1)*i*args.batch_size_prompt,...].mean)
                
        for i, input in enumerate(batched_input):
            if replace[i] == torch.tensor(True):
                input['best_image'] = input['image'].clone().detach()
                
                if args.only_attack: 
                    bloss = loss
                else:
                    bloss[i*args.batch_size_prompt:(i+1)*i*args.batch_size_prompt,...] = \
                        loss[i*args.batch_size_prompt:(i+1)*i*args.batch_size_prompt,...]

    return bloss, batched_input


def pgd_generator(batched_input, target, model, attack_type='Linf', eps=4/255, attack_steps=3, attack_lr=4/255*2/3, random_start_prob=-1e9, targeted=False, attack_criterion='regular', use_best=True, eval_mode=True):
    # generate adversarial examples
    prev_training = bool(model.training)
    if eval_mode:
        model.eval()
    
    for input in batched_input:    
        input['orig_image'] = input['image'].detach()
        input['bset_image'] = None
    assert attack_type in ['Linf', 'L2'], '{} is not supported!'.format(attack_type)
    
    if attack_type == 'Linf':
        step = LinfStep(eps=eps, orig_input=batched_input, step_size=attack_lr)
    elif attack_type == 'L2':
        step = L2Step(eps=eps, orig_input=batched_input, step_size=attack_lr)

    if attack_criterion == 'regular':
        attack_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif attack_criterion == 'smooth':
        attack_criterion = LabelSmoothingCrossEntropy()
    elif attack_criterion == 'mixup':
        attack_criterion = SoftTargetCrossEntropy()
        attack_criterion = SoftTargetCrossEntropy()

    # print("pgd one image")
    m = -1 if targeted else 1
    best_loss = None

    if random.random() < random_start_prob:
        # print("sb")
        batched_input = step.random_perturb(batched_input)

    for _ in range(attack_steps):        
        for input in batched_input:    
            input['image'] = input['image'].clone().detach().requires_grad_(True)
        
        batched_output, interm_embeddings =  model(batched_input, multimask_output=False)
        masks = torch.empty(0,1,256,256).cuda()   
        for output in batched_output:
            masks = torch.concat([masks, output['low_res_logits']])
        
        #print(masks.shape,target.shape)
        adv_losses = attack_criterion(masks, target/255.0)
        #print(adv_losses.shape)
        torch.mean(m * adv_losses).backward()
        
        for input in batched_input:
            input['grad'] = input['image'].grad.detach()
        
        with torch.no_grad():
            varlist = [adv_losses, best_loss, batched_input, m]
            best_loss, batched_input = replace_best(*varlist) 

            batched_input = step.step(batched_input)
            batched_input = step.project(batched_input)
    
    batched_output,interm_embeddings =  model(batched_input,multimask_output=False)
    masks = torch.empty(0,1,256,256).cuda()    
    for output in batched_output:
        masks = torch.concat([masks, output['low_res_logits']])   
    adv_losses = attack_criterion(masks, target/255.0)
    varlist = [adv_losses, best_loss, batched_input, m]
    best_loss, batched_input = replace_best(*varlist)
    if prev_training:
        model.train()
    
    if use_best:
        for input in batched_input:    
            input['image'] = input['best_image'].clone().detach()
    
    return batched_input

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
    parser.add_argument('--batch_size_prompt', default=4, type=int)
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


def train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    
    if args.only_attack: Path(os.path.join(args.output,'adv_examples',train_datasets[0]['name'])).mkdir(exist_ok=True, parents=True)    
    print(os.path.join(args.output,'adv_examples',train_datasets[0]['name']))
    raise NameError
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
            
            # print(type(batched_input))
            advinput = pgd_generator(batched_input, labels_256, sam, attack_type='Linf',eps=16, attack_steps=4, attack_lr=4, attack_criterion='regular' ,use_best=False, eval_mode=False)
            
            if args.only_attack:
                img_np = advinput[0]['image'].permute(1,2,0).cpu().data.numpy()
                cv2.imwrite(os.path.join(args.output,'adv_examples',train_datasets[0]['name'],ori_im_path[0].split('/')[-1]), img_np[:,:,::-1].astype(np.uint8))
                continue
            
            batched_output, interm_embeddings = sam(advinput, multimask_output=False)
    
            masks = torch.empty(0,1,256,256).cuda()
            
            for output in batched_output:
                masks = torch.concat([masks, output['low_res_logits']])
            
            loss_mask, loss_dice = loss_masks(masks, labels.unsqueeze(1)/255.0, len(masks))
            loss = loss_mask + loss_dice
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(f"Epoch {str(epoch)}: "+str(train_stats)[1:-1]+'\n')
        
        lr_scheduler.step()
        dist.barrier()
        test_stats = evaluate(args, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        sam.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(sam.module.state_dict(), args.output + model_name)
    
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

@torch.no_grad()
def evaluate(args, sam, valid_dataloaders, visualize=False):
    sam.eval()
    print("Validating...")
    test_stats = {}
    dataset_id = -1
    
    for k in range(len(valid_dataloaders)):
        dataset_id += 1 
        bad_examples = 0
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,10):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori ,ori_im_path= data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'],data_val['ori_im_path']
            K,N,H,W = labels_val.shape
            k,n,h,w = labels_ori.shape

            if n == 0:
                bad_examples += 1
                loss_dict = {"val_iou_"+str(k): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(k): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue
            
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.reshape(K*N,H,W).cuda() #K*N 1024 1024 
                labels_ori = labels_ori.reshape(k*n,h,w).cuda()
            
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy() # K 3 1024 1024 -> k 1024 1024 3
            
            if args.prompt_type=='box': 
                labels_box = misc.masks_to_boxes(labels_val) #K*N 4    
            if args.prompt_type=='point':        
                try:
                    labels_points = misc.masks_sample_points(labels_val) #[K*N 10 2]
                except:
                    bad_examples+=1
                    loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                    loss_dict_reduced = misc.reduce_dict(loss_dict)
                    metric_logger.update(**loss_dict_reduced)
                    continue
                
            batched_input = []
            dict_input = dict()
                
            input_image = torch.as_tensor(imgs[0].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # 3 1024 1024
            dict_input['image'] = input_image 
            if args.prompt_type == 'box':
                dict_input['boxes'] = labels_box #N 4
            elif args.prompt_type == 'point': 
                point_coords = labels_points #[N 10 2]
                print(point_coords.shape)
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.size()[:2], device=point_coords.device)
            elif args.prompt_type == 'noise_mask':
                dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
            else:
                raise NotImplementedError
            
            dict_input['original_size'] = imgs[0].shape[:2]
            batched_input.append(dict_input)

            batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
        
            masks = batched_output[0]['low_res_logits']

            print(masks.shape,labels_ori.shape)
            
            try:
                iou,iou_list = compute_iou(masks,labels_ori.unsqueeze(1))
                boundary_iou,boundary_iou_list = compute_boundary_iou(masks,labels_ori.unsqueeze(1))
            except:
                bad_examples += 1
                loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue

            if torch.isnan(iou).any() or torch.isnan(boundary_iou).any():
                bad_examples += 1
                loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue
            
            
            save_dir = os.path.join(args.output, args.prompt_type, valid_datasets[dataset_id]['name'])
            Path(save_dir).mkdir(parents=True,exist_ok=True)
            base = ori_im_path[0].split('/')[-1].split('.')[0]
            save_base = os.path.join(save_dir, str(base))
            record_iou(save_base, iou_list, boundary_iou_list)
            if visualize:
                masks_vis = (F.interpolate(masks.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                imgs_ii = imgs[0].astype(dtype=np.uint8)
    
                if args.prompt_type=='box':
                    show_anns(labels_val.cpu(), masks_vis, None, labels_box.cpu(), None, save_base , imgs_ii, iou_list, boundary_iou_list)
                elif args.prompt_type=='point':
                    show_anns(labels_val.cpu(), masks_vis, labels_points.cpu(), None, torch.ones(labels_points.shape[:2]).cpu(), save_base , imgs_ii, iou_list, boundary_iou_list)
            
            loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): iou, "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)            
                    
            
            print('============================')
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            test_stats.update(resstat)
            
            text_log = {k: round(meter.global_avg*100,2) for k, meter in metric_logger.meters.items() if meter.count > 0}
            if misc.is_main_process():
                with open(args.output+'/log.txt','a') as f:
                    f.write(str(valid_datasets[dataset_id]['name'])+' '+ str(text_log)[1:-1].replace("'","")+'\n')    
                    f.write(str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples) +'\n') 

    return test_stats

if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------
    
    ## Train dataset
    
    dataset_sa000000_vis_adversarial_samples = {"name": "sam_subset",
        "im_dir": "../sam-1b/sa_000000_subset",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    dataset_sa000000 = {"name": "sam_subset",
        "im_dir": "../sam-1b/sa_000000",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    dataset_sa000000_512 = {"name": "sam_subset",
        "im_dir": "../sam-1b/sa_000000/512",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    dataset_sa000000adv = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-100-1-2-10-Clip-0.2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000adv_dice = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000adv_dice_0_4 = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.4-10-0.04-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}

    dataset_sa000000adv_dice_0_8 = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.8-10-0.08-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000001adv_dice = {"name": "sam_subset",
        "im_dir": "../output/sa_000001-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000001",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000_Inversion = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Inversion/inv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000adv_1600 = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-1600.0-1-2-10-Clip-0.2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sa000000inv = {"name": "sam_subset",
        "im_dir": "../output/sa_000000@4-Grad/diversity-01-mi-SD-9.0-20-SAM-sam-vit_b-4-ADV-0.2-10-0.02-0.5-10.0-0.1-2/adv",
        "gt_dir": "../../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
            
    dataset_sam_subset_adv_vit_huge = {"name": "sam_subset",
        "im_dir": "../11187-Grad/skip-ablation-01-mi-0.5-sam-vit_h-40-0.01-100-1-2-10-Clip-0.2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ".json"}
    
    dataset_DatasetDM = {"name": "DatasetDM",
        "im_dir": "../DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/Image",
        "gt_dir": "../DatasetDM/DataDiffusion/SAM_Train_10_images_t1_10layers_NoClass_matting/label",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"}
    
    dataset_sa000000pgd = {"name": "sam_subset",
        "im_dir": "work_dirs/PGD",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ".json"}
    
    dataset_sa00000pgd_512 = {"name": "sam_subset",
        "im_dir": "work_dirs/PGD_512",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    ## valid set
    dataset_hrsod_val = {"name": "HRSOD-TE",
        "im_dir": "data/HRSOD-TE/imgs",
        "gt_dir": "data/HRSOD-TE/gts",
        "im_ext": ".jpg",
        "gt_ext": ".png"}

    
    dataset_ade20k_val = {"name": "ADE20K_2016_07_26",
        "im_dir": "data/ADE20K_2016_07_26/images/validation",
        "gt_dir": "data/ADE20K_2016_07_26/images/validation",
        "im_ext": ".jpg",
        "gt_ext": "_seg.png"}
    
    dataset_cityscapes_val = {"name": "cityscaps_val",
        "im_dir": "data/cityscapes/leftImg8bit/val",
        "gt_dir": "data/cityscapes/gtFine/val",
        "im_ext": "_leftImg8bit.png",
        "gt_ext": "_gtFine_instanceIds.png"}
    
    dataset_voc2012_val = {"name": "voc2012_val",
        "im_dir": "data/VOC2012/JPEGImages_val",
        "gt_dir": "data/VOC2012/SegmentationObject",
        "im_ext": ".jpg",
        "gt_ext": ".png"}

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
    
    dataset_ishape_antenna = {"name": "ishape",
        "im_dir": "data/ishape_dataset/antenna/val/image",
        "gt_dir": "data/ishape_dataset/antenna/val/instance_map",
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
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/test.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_train = {"name": "Plittersdorf_coco",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/train.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_val = {"name": "Plittersdorf_coco",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/val.json",
        "im_ext": ".jpg",
    }
    
        
    dataset_egohos = {"name": "egohos",
        "im_dir": "data/egohos/val/image",
        "gt_dir": "data/egohos/val/label",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_LVIS = {"name": "LVIS",
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
    
    dataset_DOORS1 = {"name": "DOORS1",
        "im_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/img",
        "gt_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS2 = {"name": "DOORS2",
        "im_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/img",
        "gt_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_NDD20_ABOVE = {"name": "NDD20",
        "im_dir": "data/NDD20/ABOVE",
        "gt_dir": "data/NDD20/ABOVE_LABELS",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_NDD20_BELOW = {"name": "NDD20_coco",
        "im_dir": "data/NDD20/BELOW",
        "annotation_file": "/data/tanglv/xhk/ASAM/2023-9-7/Ad-Sam-Main/sam_continue_learning/data/COCO2017-val/instances_val2017.json",
        "im_ext": ".jpg",
    }   
        
    dataset_PIDRAY = {"name": "pid_coco",
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
        "im_dir": "data/splits_final_deblurred/train/data",
        "gt_dir": "data/splits_final_deblurred/train/sem_seg",
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
    
    dataset_gtea_train = {"name": "gtea",
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
        "im_dir": "data/sessile-main-Kvasir-SEG/images",
        "gt_dir": "data/sessile-main-Kvasir-SEG/masks",
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
