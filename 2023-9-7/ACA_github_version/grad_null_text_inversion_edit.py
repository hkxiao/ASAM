import pdb
import os
import argparse
from get_model import get_model
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm, trange
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
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

'''
CUDA_VISIBLE_DEVICES=0 python3 grad_null_text_inversion_edit.py --model mnv2 --beta 1 --alpha 0.01 --steps 10 --norm 2
'''

############## Initialize #####################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', type=str, default='resnet50', help='cnn')
parser.add_argument('--alpha', type=float, default=0.01, help='cnn')
parser.add_argument('--beta', type=float, default=1, help='cnn')
parser.add_argument('--eps', type=float, default=0.2, help='cnn')
parser.add_argument('--steps', type=int, default=10, help='cnn')
parser.add_argument('--norm', type=int, default=2, help='cnn')
parser.add_argument('--start', default=0, type=int, help='random seed')
parser.add_argument('--end', default=1000, type=int, help='random seed')
parser.add_argument('--prefix', type=str, default='skip-ablation-01-mi', help='cnn')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mu', default=0.5, type=float, help='random seed')


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.Generator().manual_seed(args.seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing Model..')
image_size = (224, 224)
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

net = get_model(args.model)
if device == 'cuda':
    net.to(device)
    cudnn.benchmark = True
net.eval()
net.cuda()

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
                    self.attention_store[key][i] = self.attention_store[key][i] + self.step_store[key][i]
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

    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

@torch.no_grad()
def text2image_ldm_stable(
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

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, prefix='inversion'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
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
        print(len(uncond_embeddings))
        text_grad = torch.zeros_like(text_embeddings).cuda()
        print(text_grad.shape)
        latent_grad = torch.ones_like(all_latents[0]).cuda()
        print(latent_grad.shape)
        # i是倒置的了
        for i in range(len(all_latents)-1, 0, -1):
            t = times[i]
            print(i, t)
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
    label=None,
    raw_img=None
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
    success = True
    momentum = 0
    for k in range(args.steps):
        latents = adv_latents
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            # print(i, t)
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)

        image = None
        with torch.enable_grad():
            latents_last = latents.detach().clone()
            latents_last.requires_grad = True
            latents_t = (1 / 0.18215 * latents_last)
            image = model.vae.decode(latents_t)['sample']
            image = (image / 2 + 0.5)
            print(4, image.max(), image.min())
            image = limitation01(image)
            image_m = F.interpolate(image, image_size)
            print(1, image_m.max(), image_m.min())
            image_m = image_m - mean[None,:,None,None]
            image_m = image_m / std[None,:,None,None]
            outputs = net(image_m)
            _, predicted = outputs.max(1)
            if label != predicted:
                best_latent = adv_latents
                success = False
            loss_ce = torch.nn.CrossEntropyLoss()(outputs, torch.Tensor([label]).long().cuda())
            print(3, image_m.max(), image_m.min(), raw_img.max(), raw_img.min())
            loss_mse = args.beta * torch.norm(image_m-raw_img, p=args.norm).mean() # **2 / 50
            loss_mse2 = args.beta * ((image_m-raw_img)**2).mean()
            loss_mse *= loss_mse / 1024**2

            print(loss_mse.item(), loss_mse2.item())
            raise NameError
            loss = loss_ce - loss_mse
            loss.backward()
            # print(image.shape)
            # print(latents.shape, latent.shape)
            print('*' * 50)
            print('Loss', loss.item(), 'Loss_ce', loss_ce.item(), 'Loss_mse', loss_mse.item())
            print(k, 'Predicted:', label, predicted, loss.item())
            print('Grad:', latents_last.grad.min(), latents_last.grad.max())

            # print(latent.min(), latent.max())
        l1_grad = latents_last.grad / torch.norm(latents_last.grad, p=1)
        print('L1 Grad:', l1_grad.min(), l1_grad.max())
        momentum = args.mu * momentum + l1_grad
        adv_latents = adv_latents + torch.sign(momentum) * args.alpha
        noise = (adv_latents - ori_latents).clamp(-args.eps, args.eps)
        adv_latents = ori_latents + noise
        latents = adv_latents.detach()

    if success:
        best_latent = latents

    # Return Best Attack
    latents = best_latent
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        # print(i, t)
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    latents = (1 / 0.18215 * latents)
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5)
    print(4, image.max(), image.min())
    image = limitation01(image)
    image = F.interpolate(image, image_size)
    print(2, image.max(), image.min())

    image = image.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image, best_latent, success


if __name__ == '__main__':
    # 初始化
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Bug /usr/local/python/lib/python3.8/site-packages/diffusers/pipeline_utils.py
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)

    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    
    # print(ldm_stable.scheduler.alphas_cumprod)
    # print(ldm_stable.scheduler.alphas_cumprod[999])
    # print(ldm_stable.scheduler.alphas_cumprod[0])


    image_nums = 1000
    img_path = 'temp/1000/inversion'
    raw_img_path = 'third_party/Natural-Color-Fool/dataset/images'
    all_prompts = open('temp/1000/prompts.txt').readlines()
    all_latents = torch.load('temp/1000/all_latents.pth')
    all_uncons = torch.load('temp/1000/all_uncons.pth')
    all_labels = open('third_party/Natural-Color-Fool/dataset/labels.txt').readlines()

    img_list = os.listdir(img_path)
    img_list.sort()
    cnt = 0
    save_path = './temp/' + args.prefix + '-' + str(args.mu) + '-' + args.model + '-' + str(args.alpha) + '-' + str(args.beta) + '-' + str(args.norm) + '-' + str(args.steps) + '-Clip-' + str(args.eps) + '/'
    print("Save Path:", save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in trange(args.start, args.end):
        if not os.path.exists(os.path.join(save_path, img_list[i].split('.')[0]+'.png')):
            img_path = os.path.join(img_path, img_list[i])
            idx = int(img_list[i].split('.')[0]) 
            prompt = all_prompts[idx].strip()
            x_t = all_latents[idx].cuda()
            uncond_embeddings = all_uncons[idx].cuda()
            label = int(all_labels[idx].strip()) - 1
            print(idx, label)
            pil_image = Image.open(os.path.join(raw_img_path, str(idx+1)+'.png')).convert('RGB').resize(image_size)
            raw_img = (torch.tensor(np.array(pil_image), device=device).unsqueeze(0)/255.).permute(0, 3, 1, 2)
            raw_img = raw_img - mean[None,:,None,None]
            raw_img = raw_img / std[None,:,None,None]


            prompts = [prompt]
            controller = EmptyControl()
            image_inv, x_t, success = text2image_ldm_stable_last(ldm_stable, prompts, controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings, label=label, raw_img=raw_img)
            ptp_utils.view_images([image_inv[0]], prefix=os.path.join(save_path, img_list[i].split('.')[0]))
            cnt += success
            print("Acc: ", cnt, '/', (i-args.start+1))
        
        else:
            print(os.path.join(save_path, img_list[i].split('.')[0]), " has existed!")

            # break

