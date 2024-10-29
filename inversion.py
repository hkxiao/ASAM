from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm, trange
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLPipeline ,ControlNetModel
from diffusers import DDIMScheduler

import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from torch.optim.adam import Adam
from PIL import Image
import time
import json
import cv2
from lavis.models import load_model_and_preprocess
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from pycocotools import mask
import argparse
import sys
from ControlNet_last.process_feat import process_feat, clip_feat
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# model setting
parser.add_argument('--model', type=str, default='sam', help='cnn')
parser.add_argument('--model_type', type=str, default='vit_b', help='cnn')
parser.add_argument('--aigc_model_type', type=str, default='SD1.5', help='cnn')
parser.add_argument('--mask_conditioning_channels', type=int, default=3, help='cnn')
parser.add_argument('--feat_conditioning_channels', type=int, default=3, help='cnn')
parser.add_argument('--mask_control_scale', type=float, default=0.5, help='cnn')
parser.add_argument('--feat_control_scale', type=float, default=0.5, help='cnn')

# base setting
parser.add_argument('--start', default=1, type=int, help='random seed')
parser.add_argument('--end', default=11187, type=int, help='random seed')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--check_controlnet', action='store_true')
parser.add_argument('--check_inversion', action='store_true')
parser.add_argument('--check_crash', action='store_true')

# inversion setting
parser.add_argument('--inversion_type', type=str, default='null_text_inversion')
parser.add_argument('--steps', type=int, default=10, help='cnn')
parser.add_argument('--ddim_steps', default=50, type=int, help='random seed')   
parser.add_argument('--guess_mode', action='store_true')   
parser.add_argument('--guidence_scale', default=7.5, type=float, help='random seed')   

# path setting
parser.add_argument('--data_root', default='/data/tanglv/data/sam-1b/sa_000000', type=str, help='random seed')   
parser.add_argument('--save_root', default='work_dirs/sa_000000-Grad', type=str, help='random seed')   
parser.add_argument('--control_mask_dir', default='/data/tanglv/data/sam-1b/sa_000000', type=str, help='random seed')    
parser.add_argument('--control_feat_dir', default='/data/tanglv/data/sam-1b/sa_000000', type=str, help='random seed')    
parser.add_argument('--caption_path', default='/data/tanglv/data/sam-1b/sa_000000-blip2-caption.json', type=str, help='random seed')    
parser.add_argument('--mask_controlnet_path', default='ckpt/control_v11p_sd15_mask_sa000000.pth', type=str, help='random seed')
parser.add_argument('--feat_controlnet_path', default='ckpt/control_v11p_sd15_mask_sa000000.pth', type=str, help='random seed')
parser.add_argument('--sd_path', default='ckpt/control_v11p_sd15_mask_sa000000.pth', type=str, help='random seed')
args = parser.parse_args()
print(args)

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

class NullInversion:
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.num_ddim_steps = NUM_DDIM_STEPS
    
    # def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    #     prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    #     alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     beta_prod_t = 1 - alpha_prod_t
    #     pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    #     pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    #     prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    #     return prev_sample

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample, difference_scale
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, mask, feat, t, context):
        # print(mask.shape, feat.shape)
        # raise NameError
        if mask != None:
            down_block_res_samples_mask, mid_block_res_sample_mask = self.model.mask_controlnet(
                        latents,
                        t,
                        encoder_hidden_states=context,
                        controlnet_cond=mask,
                        return_dict=False,
                    )
        else: down_block_res_samples_mask, mid_block_res_sample_mask = None, None
        if down_block_res_samples_mask!=None and args.guess_mode and mid_block_res_sample_mask.shape[0]==2:
            down_block_res_samples_mask = [d[1:] for d in down_block_res_samples_mask]
            mid_block_res_sample_mask =  mid_block_res_sample_mask[1:]
            down_block_res_samples_mask = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples_mask]
            mid_block_res_sample_mask = torch.cat([torch.zeros_like(mid_block_res_sample_mask), mid_block_res_sample_mask])
            
        if feat != None:
            down_block_res_samples_feat, mid_block_res_sample_feat = self.model.feat_controlnet(
                        latents,
                        t,
                        encoder_hidden_states=context,
                        controlnet_cond=feat,
                        return_dict=False,
                    )
        else: down_block_res_samples_feat, mid_block_res_sample_feat = None, None
        if down_block_res_samples_feat!=None and args.guess_mode and mid_block_res_sample_feat.shape[0]==2:
            down_block_res_samples_feat = [d[1:] for d in down_block_res_samples_feat]
            mid_block_res_sample_feat =  mid_block_res_sample_feat[1:]
            down_block_res_samples_feat = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples_feat]
            mid_block_res_sample_feat = torch.cat([torch.zeros_like(mid_block_res_sample_feat), mid_block_res_sample_feat])                 
    
        down_block_res_samples = [down_block_res_sample_mask * args.mask_control_scale + \
                                down_block_res_sample_feat * args.feat_control_scale \
                                for (down_block_res_sample_mask, down_block_res_sample_feat) in zip(down_block_res_samples_mask, down_block_res_samples_feat)]
        mid_block_res_sample = mid_block_res_sample_mask * args.mask_control_scale + \
                                mid_block_res_sample_feat * args.feat_control_scale
         
        noise_pred = self.model.unet(
            latents, t, encoder_hidden_states=context,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, mask, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        # print(latents_input.shape, mask.shape, context.shape)
        noise_pred = self.get_noise_pred_single(latents_input, mask, t, context)
        
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    # To Do for SD-xl: 
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, mask, feat):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, mask, feat, t, cond_embeddings)
            latent = self.next_step(noise_pred ,t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, mask, feat):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent, mask, feat)
        return image_rec, ddim_latents

    def null_optimization(self, latents, mask, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, mask,t, cond_embeddings)
            for j in range(num_inner_steps):
                if args.guess_mode:
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, None, t, uncond_embeddings)
                else:
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, mask, t, uncond_embeddings)
                
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                masks = torch.cat([mask,mask])
                latent_cur = self.get_noise_pred(latent_cur, masks, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, img_path: str, mask_control: torch.tensor, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        
        image_gt =  Image.open(img_path).convert('RGB').resize((512,512))
        image_gt = np.array(image_gt).astype(np.float32)
        
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt, mask_control)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, mask_control, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale, mask, feat):
        noise_loss_list = []
        
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2)) # ([1, 4, 64, 64])
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0]) # ([1, 4, 64, 64])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():                
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), mask, feat, t, self.context) # ([2, 4, 64, 64])
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
        return noise_loss_list
    
    def direct_invert(self, img_path, mask_control, feat_control, prompt, guidance_scale=7.5, num_inner_steps=10, early_stop_epsilon=1e-5):
        
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        
        image_gt =  Image.open(img_path).convert('RGB').resize((512,512))
        image_gt = np.array(image_gt).astype(np.float32)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt, mask=mask_control, feat=feat_control)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale, mask_control, feat_control)
        return (image_gt, image_rec), ddim_latents[-1], noise_loss_list
            

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    mask_control: Optional[torch.Tensor] = None,
    feat_control: Optional[torch.Tensor] = None,
    uncond_embeddings=None,
    noise_loss_list=None,
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

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])

        latents = ptp_utils.diffusion_step(model, controller, latents, mask_control, feat_control, context, t, guidance_scale, \
            low_resource=False, guess_mode=args.guess_mode, mask_control_scale=args.mask_control_scale, feat_control_scale=args.feat_control_scale)
        if noise_loss_list: latents = latents + noise_loss_list[i]
        #print(torch.max(context), torch.min(context), torch.max(latents),  torch.min(latents), torch.max(noise_loss_list[i]), torch.min(noise_loss_list[i]))
    
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(prompts, controller, latent=None, mask_control=None, feat_control=None, run_baseline=False, generator=None, uncond_embeddings=None, noise_loss_list=None, verbose=True, prefix='inversion'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, mask_control=mask_control, feat_control=feat_control, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, mask_control=mask_control, feat_control=feat_control, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, noise_loss_list=noise_loss_list)
    if verbose:
        ptp_utils.view_images(images, prefix=prefix)
    return images, x_t

def check_controlnet():
    id = 1    
    prompt = captions[f'sa_{str(id)}.jpg']

    control_image = Image.open(os.path.join(args.control_mask_dir, f'sa_{str(id)}.png'))
    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512,512)) / 255.0
    
    output = ldm_stable(
    "",  image=control_image,num_inference_steps=50, guidance_scale=1.0
    ).images[0]
    
    output_numpy = np.array(output)
    print(type(output),output_numpy.max())
    output.save('check_controlnet_pth.png')
    
    controller = AttentionStore()
    x_t = torch.randn((1, ldm_stable.unet.in_channels,  512 // 8, 512 // 8))
    run_and_display(prompts=[prompt], controller=controller, run_baseline=False, latent=x_t, mask_control=control_image,uncond_embeddings=None, verbose=True, prefix='check_controlnet_use')

def check_inversion():
    id = 1    
    prompt = captions[f'sa_{str(id)}.jpg']

    latent_path = f"{args.inversion_dir}/sa_{str(id)}_latent.pth"
    uncond_path = f"{args.inversion_dir}/sa_{str(id)}_uncond.pth"
    x_t = torch.load(latent_path).cuda()
    uncond_embeddings = torch.load(uncond_path).cuda()
    
    control_image = Image.open(os.path.join(args.control_mask_dir, f'sa_{str(id)}.png'))
    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512,512)) / 255.0
    control_image = torch.from_numpy(control_image).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
    controller = AttentionStore()
    
    run_and_display(prompts=[prompt], controller=controller, run_baseline=False, latent=x_t, mask_control=control_image,uncond_embeddings=uncond_embeddings, verbose=True, prefix='check_inversion')

def check_crash():
    # Inversion loop
    for i in range(args.start, args.end+1):
        if not os.path.exists(os.path.join(args.save_root, 'embeddings', 'sa_'+str(i)+'_latent.pth')) and \
         os.path.exists(os.path.join(args.data_root, 'sa_'+str(i)+'.jpg')):
            sys.exit(-1)
    sys.exit(0)
    
if __name__ == '__main__':
    if args.check_crash: check_crash()
    
    # Load Stable Diffusion 
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = args.ddim_steps
    GUIDANCE_SCALE = args.guidence_scale
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if 'pth' in args.mask_controlnet_path : mask_controlnet = ControlNetModel.from_single_file(args.mask_controlnet_path).to(device)
    else: mask_controlnet = ControlNetModel.from_pretrained(args.mask_controlnet_path).to(device)

    if 'pth' in args.feat_controlnet_path : feat_controlnet = ControlNetModel.from_single_file(args.feat_controlnet_path, conditioning_channels=args.feat_conditioning_channels).to(device)
    else: feat_controlnet = ControlNetModel.from_pretrained(args.feat_controlnet_path).to(device)

    ldm_stable = StableDiffusionControlNetPipeline.from_pretrained(args.sd_path, use_auth_token=MY_TOKEN,controlnet=mask_controlnet, scheduler=scheduler).to(device)
    ldm_stable.feat_controlnet = feat_controlnet
    ldm_stable.mask_controlnet = mask_controlnet
    
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    if args.aigc_model_type == 'SDXL': tokenizer_2 = ldm_stable.tokenizer_2
    
    inversion = NullInversion(ldm_stable)
    
    # Load Caption
    captions = {}
    with open(args.caption_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dict = json.loads(line.strip()) 
            captions[json_dict['img'].strip()] = json_dict['prompt'].strip()   
            
    # Check controlnet & inversion
    if args.check_controlnet: check_controlnet()
    if args.check_inversion: check_inversion()
    if args.check_inversion or args.check_controlnet: raise NameError 
    
    # Prepare save dir
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)
    if not os.path.exists(os.path.join(args.save_root , 'pair')): os.mkdir(os.path.join(args.save_root , 'pair'))
    if not os.path.exists(os.path.join(args.save_root , 'inv')): os.mkdir(os.path.join(args.save_root , 'inv'))
    if not os.path.exists(os.path.join(args.save_root , 'embeddings')): os.mkdir(os.path.join(args.save_root , 'embeddings'))
    
    # names = sorted(os.listdir(args.data_root))
    # names = [name[:-4] for name in names if '.jpg' in name]
    # if args.end != -1:
    #     names = names[args.start:args.end+1]
    # else:
    #     names = names[args.start:]
    
    # print(names)    
    # Adversarial optimization loop
    # for name in tqdm(names):

    for i in range(args.start, args.end):
        name = 'sa_' + str(i)
        img_path = args.data_root+'/'+name+'.jpg'
        control_mask_path = args.control_mask_dir+'/'+name+'.png'
        control_sd_feat_path = args.control_feat_dir+'/'+name+'_sd.pth'
        control_dino_feat_path = args.control_feat_dir+'/'+name+'_dino.pth'
        
        # prepare img & control mask path
        # img_path = os.path.join(args.data_root, name + '.jpg')
        # control_mask_path = os.path.join(args.control_mask_dir, name + '.png')
        
        if not os.path.exists(img_path):
            print(img_path, "does not exist!")
            continue
        
        # init x_t & uncond_emb
        latent_path = f"{args.save_root}/embeddings/{name}_latent.pth"
        uncond_path = f"{args.save_root}/embeddings/{name}_uncond.pth"
        if os.path.exists(latent_path) and os.path.exists(uncond_path):
            print(latent_path, uncond_path, " has existed!")
            #continue
                
        # load catpion
        if img_path.split('/')[-1] in captions:
            prompt = captions[img_path.split('/')[-1]]
        else: 
            print("DCI don't have caption for", img_path.split('/')[-1])
            continue
        print(prompt)
        
        # load control mask
        mask_control = cv2.imread(control_mask_path).astype(np.float32)
        mask_control = cv2.cvtColor(mask_control, cv2.COLOR_BGR2RGB)
        mask_control = cv2.resize(mask_control, (512,512)) / 255.0
        mask_control = torch.from_numpy(mask_control).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
        
        # load control feat
        sd_feat = torch.load(control_sd_feat_path, map_location='cuda')
        dino_feat = torch.load(control_dino_feat_path, map_location='cuda')

        # process control feat
        feat = process_feat(sd_feat, dino_feat, sd_target_dim=[4,4,4], dino_target_dim=12, dino_pca=True, using_sd=True, using_dino=True) #[1 C H W]
        feat = feat.flatten(-2).permute(0,2,1).unsqueeze(0) # (1,1,H*W,C)
        feat = clip_feat(feat, img_path = img_path) #[H W C]
        feat = feat.permute(2,0,1).unsqueeze(0) #[1 C H W]
        feat = F.interpolate(feat, size=(512,512), mode='bilinear', align_corners=False).permute(0,2,3,1).squeeze(0) #[H W C]
        feat_min = feat.view(512*512,-1).min(dim=0,keepdim=True)[0].view(1,1,-1)
        feat_max = feat.view(512*512,-1).max(dim=0,keepdim=True)[0].view(1,1,-1)
        feat_control = (feat - feat_min) / (feat_max - feat_min)
        feat_control = feat_control.permute(2,0,1).unsqueeze(0)
            
        # inversion
        uncond_embeddings, noise_loss_list = None, None
        start = time.time()
        if args.inversion_type == 'null_text_inversion':
            (image_gt, image_enc), x_t, uncond_embeddings = inversion.invert(img_path, mask_control=mask_control, num_inner_steps=args.steps, prompt = prompt, offsets=(0,0,0,0), verbose=True)
        elif args.inversion_type == 'direct_inversion':
            (image_gt, image_rec), x_t, noise_loss_list = inversion.direct_invert(img_path, mask_control=mask_control, feat_control=feat_control, num_inner_steps=args.steps, prompt = prompt)        
        
        print('Inversion Time:', time.time() - start)
        
        if uncond_embeddings: gather_uncond_embeddings = torch.cat(uncond_embeddings, 0)        
        if noise_loss_list: gather_noise_loss = torch.cat(noise_loss_list, 0)
        
        # save x_t & uncond_emb
        torch.save(x_t, f'{args.save_root}/embeddings/{name}_latent.pth')
        if uncond_embeddings: torch.save(gather_uncond_embeddings, f'{args.save_root}/embeddings/{name}_uncond.pth')
        if noise_loss_list: torch.save(gather_noise_loss, f'{args.save_root}/embeddings/{name}_noise_loss.pth')
        
        # show 
        controller = AttentionStore()
        image_inv, x_t = run_and_display(prompts=[prompt], controller=controller, run_baseline=False, latent=x_t, mask_control=mask_control, feat_control=feat_control, uncond_embeddings=uncond_embeddings, noise_loss_list=noise_loss_list,verbose=False)
        ptp_utils.view_images([image_gt, image_inv[0]], prefix=f'{args.save_root}/pair/{name}', shuffix='.jpg')
        ptp_utils.view_images([image_inv[0]], prefix=f'{args.save_root}/inv/{name}', shuffix='.jpg')

       
