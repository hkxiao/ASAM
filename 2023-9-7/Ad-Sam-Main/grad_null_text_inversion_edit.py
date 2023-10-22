import pdb
import os
import argparse
from get_model import get_model
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm, trange
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler
from diffusers import ControlNetModel
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import time
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import json    
from pycocotools import mask
from sam_continue_learning.segment_anything.MyPredictor import SamPredictor

'''
CUDA_VISIBLE_DEVICES=0 python3 grad_null_text_inversion_edit.py --model sam --beta 1 --alpha 0.01 --steps 10  --ddim_steps=50 --norm 2
'''

############## Initialize #####################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', type=str, default='sam', help='cnn')
parser.add_argument('--model_type', type=str, default='vit_b', help='cnn')
parser.add_argument('--alpha', type=float, default=0.01, help='cnn')
parser.add_argument('--gamma', type=float, default=100, help='cnn')
parser.add_argument('--beta', type=float, default=1, help='cnn')
parser.add_argument('--eps', type=float, default=0.2, help='cnn')
parser.add_argument('--steps', type=int, default=10, help='cnn')
parser.add_argument('--norm', type=int, default=2, help='cnn')
parser.add_argument('--sam_batch', type=int, default=150, help='cnn')
parser.add_argument('--start', default=1, type=int, help='random seed')
parser.add_argument('--end', default=11187, type=int, help='random seed')
parser.add_argument('--prefix', type=str, default='skip-ablation-01-mi', help='cnn')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mu', default=0.5, type=float, help='random seed')
parser.add_argument('--mae_thd', default=0.8, type=float, help='random seed')
parser.add_argument('--ddim_steps', default=50, type=int, help='random seed')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.Generator().manual_seed(args.seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing Model..')
image_size = (1024, 1024)

if args.model == 'vit' or args.model == 'adv_resnet152_denoise':
    print('Using 0.5 Nor...')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
elif args.model == 'mvit':
    mean = [0, 0, 0]
    std = [1, 1, 1] 
    image_size = (320, 320)
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

mean = torch.Tensor(mean).cuda()
std = torch.Tensor(std).cuda()

net = get_model(args.model, args.model_type)
if device == 'cuda':
    net.to(device)
    cudnn.benchmark = True
net.eval()
net.cuda()

if args.model == 'sam':
    net = SamPredictor(net)

class LocalBlend:
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


def mae_value(x, y):
    return torch.abs(x-y).mean()

@torch.no_grad()
def compute_iou(preds, target):
    def mask_iou(pred_label,label):
        '''
        calculate mask iou for pred_label and gt_label
        '''

        pred_label = (pred_label>0)[0].int()
        label = (label>0.5)[0].int()

        intersection = ((label * pred_label) > 0).sum()
        union = ((label + pred_label) > 0).sum()
        return intersection / union
    
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)
    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), prefix='cross_attention')
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), prefix='self_attention')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    # 抠出一个正方形，会裁边
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    # ttt = Image.fromarray(image)
    # ttt.save('temp/inversion/mid.jpg')
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    mask_control: Optional[torch.tensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    # print(batch_size)
    # raise NameError
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, mask_control, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def run_and_display(prompts, controller, latent=None, mask_control=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, prefix='inversion'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, mask_control=mask_control, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, mask_control=mask_control,num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images, prefix=prefix)
    return images, x_t

@torch.enable_grad()
def text2image_ldm_stable_grad(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, init_latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    all_latents = []
    times = []
    latents = init_latents
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        print(i, t, len(all_latents))
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        all_latents.append(latents)
        times.append(t)

    with torch.no_grad():
        if return_type == 'image':
            image = ptp_utils.latent2image(model.vae, latents)
        else:
            image = latents

    # 开始反传
    with torch.enable_grad():
        #print(len(uncond_embeddings))
        text_grad = torch.zeros_like(text_embeddings).cuda()
        #print(text_grad.shape)
        latent_grad = torch.ones_like(all_latents[0]).cuda()
        #print(latent_grad.shape)
        # i是倒置的了
        for i in range(len(all_latents)-1, 0, -1):
            t = times[i]
            #print(i, t)
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            grad_latents = all_latents[i].clone().detach()
            grad_latents.requires_grad = True
            print(grad_latents.requires_grad, grad_latents.is_leaf)
            # test_cat = torch.cat([all_latents[i]] * 2)
            # print("Test Cat: ", test_cat.shape, test_cat.requires_grad, test_cat.grad_fn)
            with torch.autograd.set_detect_anomaly(True):
                temp_latents = diffusion_step_grad(model, controller, grad_latents, context, t, guidance_scale, low_resource=False)
                print(temp_latents.requires_grad)
                # temp_latents.backward(latent_grad)
                # print(grad_latents.grad)
            break

    return image, latent, all_latents

@torch.enable_grad()
def diffusion_step_grad(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        print("Latent: ", latents.shape, latents.requires_grad)
        latents_input = torch.cat([latents] * 2)
        print("Latent input: ", latents_input.shape, latents_input.requires_grad, latents_input.grad_fn)
        #### 修改 /usr/local/python/lib/python3.8/site-packages/diffusers
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        print("Noise_pred: ", noise_pred.shape, noise_pred.requires_grad)
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        print(noise_pred_uncond.requires_grad, noise_prediction_text.requires_grad)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # noise_pred.sum().backward()
    # print(latents.grad)
    latents_step = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents_step.sum().backward()
    print(latents.grad)
    latents_new = controller.step_callback(latents_step)
    print((latents_new-latents).abs().sum())
    return latents_new

@torch.no_grad()
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

def limitation01(y):
    idx = (y > 1)
    y[idx] = (torch.tanh(1000*(y[idx]-1))+10000)/10001
    idx = (y < 0)
    y[idx] = (torch.tanh(1000*(y[idx])))/10000
    return y


@torch.no_grad()
def text2image_ldm_stable_last(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    boxes=None,
    mask_label=None,
    mask_control=None,
    raw_img=None,
    predicted_iou=None,
    annotations=None,
    
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    print("Latent", latent.shape, "Latents", latents.shape)
    model.scheduler.set_timesteps(num_inference_steps)

    best_latent = latents
    ori_latents = latents.clone().detach()
    adv_latents = latents.clone().detach()
    print(latents.max(), latents.min())
    momentum = 0
    worst_iou = 1.0
    worst_mask = None
    
    origin_width, origin_height = info_dict['image']['width'], info_dict['image']['height']
    boxes = torch.empty(0,4).cuda()
    mask_labels = torch.empty(0,1,256,256).cuda() 
    for i, annotation in enumerate(annotations):
        encode_mask = annotation['segmentation']
        decoded_mask = mask.decode(encode_mask)
        mask_label = cv2.resize(decoded_mask, (256,256))
        mask_label = torch.from_numpy(mask_label).to(torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        mask_labels = torch.concat([mask_labels, mask_label])
        
        x, y, w, h = annotation['bbox']
        x_min, y_min = 1024 * x / origin_width, 1024 * y / origin_height
        x_max, y_max = 1024 * (x + w) / origin_width, 1024 * (y + h) / origin_height
        box =  torch.tensor([[x_min, y_min, x_max, y_max]], device=device, dtype=torch.float32)
        boxes = torch.concat([boxes, box]) 
    
    for k in range(args.steps):
        latents = adv_latents
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            # print(i, t)
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, mask_control, context, t, guidance_scale, low_resource=False)

        image = None
        with torch.enable_grad():
            latents_last = latents.detach().clone()
            latents_last.requires_grad = True
            latents_t = (1 / 0.18215 * latents_last)
            image = model.vae.decode(latents_t)['sample']
            image = (image / 2 + 0.5)
            #print(4, image.max(), image.min())
            image = limitation01(image)
            image_m = F.interpolate(image, image_size)
            #print(1, image_m.max(), image_m.min())
            net.set_torch_image(image_m*255.0,original_image_size=(1024,1024))
                        
            ad_masks, ad_iou_predictions, ad_low_res_logits = net.predict_torch(point_coords=None,point_labels=None,boxes=boxes,multimask_output=False)
            loss_ce = args.gamma * torch.nn.functional.binary_cross_entropy_with_logits(ad_low_res_logits,mask_labels) 
        
            iou = compute_iou(ad_low_res_logits, mask_labels).item()
            if iou < worst_iou:                
                best_latent, worst_iou, worst_mask  = adv_latents, iou, F.interpolate(ad_masks.to(torch.float32), size=(512,512), mode='bilinear', align_corners=False)
                    
            image_m = image_m - mean[None,:,None,None]
            image_m = image_m / std[None,:,None,None]
            print(k, image_m.max(), image_m.min(), raw_img.max(), raw_img.min())
            loss_mse =  args.beta * torch.norm(image_m-raw_img, p=args.norm).mean()  # **2 / 50
            #loss_mse = -args.beta * ((image_m-raw_img)**2).mean()
            
            loss = loss_ce - loss_mse
            loss.backward()
            print('*' * 50)
            print('Loss', loss.item(),'Loss_ce', loss_ce.item(), 'Loss_mse', loss_mse.item())
            print(k, 'Predicted:', loss)
            print('Grad:', latents_last.grad.min(), latents_last.grad.max())
            # print(latent.min(), latent.max())
        
        l1_grad = latents_last.grad / torch.norm(latents_last.grad, p=1)
        print('L1 Grad:', l1_grad.min(), l1_grad.max())
        momentum = args.mu * momentum + l1_grad
        adv_latents = adv_latents + torch.sign(momentum) * args.alpha
        noise = (adv_latents - ori_latents).clamp(-args.eps, args.eps)
        adv_latents = ori_latents + noise
        latents = adv_latents.detach()

    # Return Best Attack
    latents = best_latent
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        # print(i, t)
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, mask_control,context, t, guidance_scale, low_resource=False)
        
    latents = (1 / 0.18215 * latents)
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5)
    print(4, image.max(), image.min())
    image = limitation01(image)
    print(2, image.max(), image.min())

    image = image.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    
    worst_mask_show = np.zeros((512,512,3))
    if args.steps:
        num = origin_len
        length = 1<<24
        for i, single_mask in enumerate(worst_mask):
            single_mask = single_mask[0].cpu().numpy()
            pos = (length-1) *(i+1) / num
            color = (pos%256, pos//256%256, pos//(1<<16))
            worst_mask_show[single_mask!=0] = color
    
    return image, best_latent, worst_mask_show, worst_iou

def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector,  8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector

def str2img(value):
    width, height = 512, 512
    background_color = (255, 255, 255)  # 白色背景
    image = cv2.UMat(np.ones((height, width, 3), dtype=np.uint8) * background_color)
    
    # 在图像上绘制文本
    text = "worst_iou: " + str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_color = (0, 0, 0)  # 黑色文本颜色
    thickness = 1

    # 获取文本的尺寸
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2  # 计算文本的起始x坐标
    y = (height + text_size[1]) // 2  # 计算文本的起始y坐标

    # 在图像上绘制文本
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

    return image.get()


if __name__ == '__main__':
    # 初始化
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = args.ddim_steps
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    controlnet = ControlNetModel.from_single_file("ckpt/control_v11p_sd15_mask_sam_v2.pth").to(device)    
    ldm_stable = StableDiffusionControlNetPipeline.from_pretrained("ckpt/stable-diffusion-v1-5", use_auth_token=MY_TOKEN,controlnet=controlnet, scheduler=scheduler).to(device)

    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    
    annotation_dir = '/data/tanglv/data/sam-1b-subset'
    dataset_dir = 'sam-subset-11187'
    
    save_path = './11187-Grad/' + args.prefix + '-' + str(args.mu) + '-' + args.model + '-' + args.model_type +'-'+ str(args.sam_batch)+ '-' +str(args.alpha) + '-' + str(args.gamma) + '-' + str(args.beta) + '-' + str(args.norm) + '-' + str(args.steps) + '-Clip-' + str(args.eps) + '/'
    print("Save Path:", save_path)
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not os.path.exists(save_path+'pair/'): os.mkdir(save_path+'pair/')
    if not os.path.exists(save_path+'adv/'): os.mkdir(save_path+'adv/')
    if not os.path.exists(save_path+'record/'): os.mkdir(save_path+'record/')
    
    captions = {}
    with open('/data/tanglv/data/sam-1b-subset-blip2-caption.json','r') as f:
        lines = f.readlines()
        for line in lines:
            json_dict = json.loads(line.strip()) 
            captions[json_dict['img'].strip()] = json_dict['prompt'].strip()   
    
    for i in trange(args.start, args.end+1):
        img_path = dataset_dir+'/'+'sa_'+str(i)+'.jpg'
        mask_path = dataset_dir+'/'+'sa_'+str(i)+'.png'
        json_path = annotation_dir+'/'+'sa_'+str(i)+'.json'
        if not os.path.exists(img_path):
            print(img_path, "does not exist!")
            continue
        
        pil_image = Image.open(img_path).convert('RGB').resize(image_size)
        raw_img_show = np.array(pil_image.resize((512,512)))
        raw_img = (torch.tensor(np.array(pil_image).astype(np.float32), device=device).unsqueeze(0)/255.).permute(0,3, 1, 2)
        raw_img = raw_img - mean[None,:,None,None]
        raw_img = raw_img / std[None,:,None,None]
        
        global info_dict
        info_dict = json.loads(open(json_path).read())
        annotations = info_dict['annotations']
        annotations = sorted(annotations, key=lambda x: x['bbox'][2]*x['bbox'][3], reverse=True)
        global origin_len
        origin_len = len(annotations)

        if len(annotations) > args.sam_batch:
            annotations = annotations[:args.sam_batch]
        
        print("========> batchsize:",len(annotations))
        
        prompt = captions[img_path.split('/')[-1]]
        print(prompt)
        
        latent_path = f"11187-Inversion/embeddings/sa_{i}_latent.pth"
        uncond_path = f"11187-Inversion/embeddings/sa_{i}_uncond.pth"
        if not os.path.exists(latent_path) or not os.path.exists(uncond_path):
            print(latent_path, uncond_path, "do not exist!")
            continue
        else:
            x_t = torch.load(latent_path).cuda()
            uncond_embeddings = torch.load(uncond_path).cuda()
        
        if os.path.exists(os.path.join(save_path, 'adv', 'sa_'+str(i)+'.png')):
            print(os.path.join(save_path, 'adv', 'sa_'+str(i)+'.png'), " has existed!")
            continue
            
        mask_control = cv2.imread(mask_path)
        mask_control = cv2.cvtColor(mask_control, cv2.COLOR_BGR2RGB)
        mask_control = cv2.resize(mask_control, (512,512))
        mask_show = mask_control.copy()
        mask_control = torch.from_numpy(mask_control).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda() / 255.0
        
        controller = EmptyControl()
        
        start = time.time()            
        image_inv, x_t, worst_mask, worst_iou = text2image_ldm_stable_last(ldm_stable, [prompt], controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings, mask_control=mask_control,raw_img=raw_img,annotations=annotations)    
        print('Generate Time:', time.time() - start)
        
        ptp_utils.view_images([image_inv[0]], prefix=os.path.join(save_path,'adv','sa_'+str(i)))
        
        ptp_utils.view_images([raw_img_show, mask_show, image_inv[0], worst_mask, str2img(worst_iou)], prefix=os.path.join(save_path,'pair','sa_'+str(i)))
        
        with open(save_path+'/record/sa_'+str(i)+'.txt','w') as f:
            f.write(str(worst_iou))