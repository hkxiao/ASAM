#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
python null_text_inversion.py \
    --save_root=output/lecun-Inversion \
    --data_root=lecun_photo \
    --control_mask_dir=lecun_photo \
    --caption_path=lecun_photo/blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa_000002_lecun_finetune.pth \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=0 \

