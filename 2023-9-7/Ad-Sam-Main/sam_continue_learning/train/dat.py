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
from torch.optim.lr_scheduler import LambdaLR
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from easyrobust.easyrobust.third_party.vqgan import reconstruct_with_vqgan, VQModel
from PIL import Image

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
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)

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
        replace = m * bloss < m * loss
        print(replace.shape)
        raise NameError
        for i, input in enumerate(batched_input):
            if replace[i] == True:
                bbatched_input[i] = input.clone().detach()
        bloss[replace] = loss[replace]

    return bloss, batched_input


def pgd_generator(batched_input, target, model, attack_type='Linf', eps=4/255, attack_steps=3, attack_lr=4/255*2/3, random_start_prob=0.0, targeted=False, attack_criterion='regular', use_best=True, eval_mode=True):
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

    m = -1 if targeted else 1
    best_loss = None

    if random.random() < random_start_prob:
        batched_input = step.random_perturb(batched_input)

    for _ in range(attack_steps):        
        for input in batched_input:    
            input['image'] = input['image'].clone().detach().requires_grad_(True)
        
        batched_output, interm_embeddings =  model(batched_input, multimask_output=False)
        masks = torch.empty(0,1,256,256).cuda()   
        for output in batched_output:
            masks = torch.concat([masks, output['low_res_logits']])
        
        adv_losses = attack_criterion(masks, target/255.0)
        torch.mean(m * adv_losses).backward()
        
        for input in batched_input:
            input['grad'] = input['image'].grad.detach()
        
        with torch.no_grad():
            varlist = [adv_losses, best_loss, batched_input, m]
            replace_best(*varlist)

            batched_input = step.step(batched_input)
            batched_input = step.project(batched_input)
    
    batched_output,interm_embeddings =  model(batched_input,multimask_output=False)
    masks = torch.empty(0,1,256,256).cuda()    
    for output in batched_output:
        masks = torch.concat([masks, output['low_res_logits']])   
    adv_losses = attack_criterion(masks, target/255.0)
    varlist = [adv_losses, best_loss, batched_input, m]
    replace_best(*varlist)
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

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
      
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('Tune-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=10, type=int)
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--slow_start', action='store_true')
    parser.add_argument('--attack_criterion', default='regular')
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_prompt_start', default=0, type=int)
    parser.add_argument('--batch_size_prompt', default=-1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--numworkers', type=int, default=-1)
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    parser.add_argument('--train-datasets', nargs='+')
    parser.add_argument('--valid-datasets', nargs='+')
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
    
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
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
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
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
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
            lr_scheduler.last_epoch = args.start_epoch
        else:
            print("slow start & fast decay")
            lr_scheduler = LambdaLR(optimizer, lr_lambda)

        train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:    
        evaluate(args, sam, valid_dataloaders, args.visualize)


def train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    ddconfig = {'double_z': False, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1,2,2,4], 'num_res_blocks': 2, 'attn_resolutions':[32], 'dropout': 0.0}
    vqgan_aug = VQModel(ddconfig, n_embed=16384, embed_dim=4, ckpt_path='http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/vqgan_openimages_f8_16384.ckpt')
    vqgan_aug = vqgan_aug.cuda()
    vqgan_aug.eval()
    
    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
 
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
    
        for data in metric_logger.log_every(train_dataloaders,10):
            
            inputs, labels = data['image'], data['label']  # [K 3 1024 1024]   [K N 1024 1024]
            
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
            
            
            with torch.no_grad():
                imgs_rec = reconstruct_with_vqgan(imgs_256/255, vqgan_aug) * 255.0
                
                # print(imgs_256.max())
                # img_256 = imgs_256[0].permute(2,1,0).cpu().numpy()
                # cv2.imwrite('ori.png',np.flip(img_256,-1))
                
                # print(imgs_rec.max())
                # img_rec = imgs_rec[0].permute(2,1,0).cpu().numpy()
                # cv2.imwrite('rec.png',np.flip(img_rec,-1))
            
            imgs_rec = F.interpolate(imgs_rec,size=(1024,1024),mode='bilinear',align_corners=False)
            for i,input in enumerate(batched_input):
                input['image']=imgs_rec[i] 
            
            advinput = pgd_generator(batched_input, labels_256, sam,attack_type='L2',eps=4, attack_steps=1, attack_lr=4*2  ,attack_criterion=args.attack_criterion, random_start_prob=0.8 ,use_best=False,eval_mode=False)
            
            imgs_adv = torch.empty(0,3,1024,1024).cuda()
            for input in advinput:
                imgs_adv = torch.concat([imgs_adv, input['image'].unsqueeze(0)])
            imgs_adv = F.interpolate(imgs_adv,size=(256,256),mode='bilinear',align_corners=False)
            
            with torch.no_grad():
                adv_imgs_rec = reconstruct_with_vqgan(imgs_adv/255, vqgan_aug) * 255.0
                # print(imgs_adv.max())
                # img_adv = imgs_adv[0].permute(2,1,0).cpu().numpy()
                # cv2.imwrite('adv.png',np.flip(img_adv,-1))
                
                #print(adv_imgs_rec.max())
                adv_img_rec = adv_imgs_rec[0].permute(2,1,0).cpu().numpy()
                cv2.imwrite('adv_rec.png',np.flip(adv_img_rec,-1))
    
            adv_imgs_rec = F.interpolate(adv_imgs_rec,size=(1024,1024),mode='bilinear',align_corners=False)
            for i,input in enumerate(advinput):
                input['image']=adv_imgs_rec[i] 
                    
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
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

@torch.no_grad()
def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

@torch.no_grad()
def evaluate(args, sam, valid_dataloaders, visualize=False):
    sam.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,10):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            K,N,H,W = labels_val.shape
            k,n,h,w = labels_ori.shape
            
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.reshape(K*N,H,W).cuda() #K*N 1024 1024 
                labels_ori = labels_ori.reshape(k*n,h,w).cuda()
            
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy() # K 3 1024 1024 -> k 1024 1024 3
            
            labels_box = misc.masks_to_boxes(labels_val) #K*N 4
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # 3 1024 1024
                dict_input['image'] = input_image.to(torch.float32) 
                input_type = random.choice(input_keys)
                sparse_slice, dense_slice = slice(b_i*N,b_i*N+N),slice(b_i*N,b_i*N+N)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[sparse_slice,...] #N 4
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
        
            masks = batched_output[0]['low_res_logits']

            iou = compute_iou(masks,labels_ori.unsqueeze(1))
            boundary_iou = compute_boundary_iou(masks,labels_ori.unsqueeze(1))
            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_vis = (F.interpolate(masks.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                       

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(str(misc.get_world_size()*len(valid_dataloader))+' '+ str(resstat)[1:-1]+'\n')    

    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------
    dataset_sam_subset_ori = {"name": "sam_subset",
                "im_dir": "../../sam-1b/sa_000000",
                "gt_dir": "../../sam-1b/sa_000000",
                "im_ext": ".jpg",
                "gt_ext": ""}
    
    dataset_sam_subset_ori_low = {"name": "sam_subset",
            "im_dir": "../../sam-1b/sa_000000/512",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".jpg",
            "gt_ext": ""}
    
    dataset_sam_subset_adv = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-100-1-2-10-Clip-0.2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sam_subset_adv_dice = {"name": "sam_subset",
        "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sam_subset_sa_000000_Inversion = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Inversion/inv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sam_subset_adv_1600 = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-1600.0-1-2-10-Clip-0.2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".png",
            "gt_ext": ""}
    
    dataset_sam_subset_div = {"name": "sam_subset",
        "im_dir": "../../output/sa_000000@4-Grad/diversity-01-mi-SD-9.0-20-SAM-sam-vit_b-4-ADV-0.2-10-0.02-0.5-10.0-0.1-2/adv",
        "gt_dir": "/data/tanglv/data/sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    dataset_sam_subset_adv_4 = {"name": "sam_subset",
            "im_dir": "../../output/sa_000000@4-Grad/skip-ablation-01-mi-SD-9.0-20-SAM-sam-vit_b-4-ADV-0.2-10-0.02-0.5-10.0-0.1-2/adv",
            "gt_dir": "../../sam-1b/sa_000000",
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
    
    dataset_sam_subset_pgd = {"name": "sam_subset",
            "im_dir": "../../data/tanglv/xhk/Ad-Sam-Main/sam_continue_learning/train/work_dirs/PGD",
            "gt_dir": "../../sam-1b/sa_000000",
            "im_ext": ".jpg",
            "gt_ext": ".json"}
    
    dataset_sam_subset_pgd_512 = {"name": "sam_subset",
        "im_dir": "work_dirs/PGD_512",
        "gt_dir": "../../sam-1b/sa_000000",
        "im_ext": ".jpg",
        "gt_ext": ""}
    
    # valid set
    
    # single
    dataset_hrsod_val = {"name": "HRSOD-TE",
            "im_dir": "../data/HRSOD-TE/imgs",
            "gt_dir": "../data/HRSOD-TE/gts",
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
            "im_dir": "../data/cityscapes/leftImg8bit/val",
            "gt_dir": "../data/cityscapes/gtFine/val",
            "im_ext": "_leftImg8bit.png",
            "gt_ext": "_gtFine_instanceIds.png"}
    #实列分割
    dataset_voc2012_val = {"name": "voc2012_val",
            "im_dir": "../data/VOC2012/JPEGImages_val",
            "gt_dir": "../data/VOC2012/SegmentationObject",
            "im_ext": ".jpg",
            "gt_ext": ".png"}
    #实列分割
    dataset_coco2017_val = {"name": "coco2017_val",
            "im_dir": "../data/COCO2017-val/val2017",
            "annotation_file": "../data/COCO2017-val/instances_val2017.json",
            "im_ext": ".jpg"
            }
    
    
    args = get_args_parser()
  
    
    train_datasets = [globals()[dataset] for dataset in args.train_datasets]
    
    # print(train_datasets)
    # raise NameError
    valid_datasets = [globals()[dataset] for dataset in args.valid_datasets]
    
    main(train_datasets, valid_datasets, args)
