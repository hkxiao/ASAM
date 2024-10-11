#!/bin/bash


export CUDA_VISIBLE_DEVICES=7
python grad_null_text_inversion_edit.py \
    --save_root=output/lecun-Grad \
    --data_root=lecun_photo \
    --control_mask_dir=lecun_photo  \
    --caption_path=lecun_photo/blip2-caption.json\
    --inversion_dir=output/lecun-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa_000002_lecun_finetune.pth \
    --prompt_bs=140 \
    --prompt_type=point \
    --prompt_num=10 \
    --eps=0.2 \
    --steps=10 \
    --alpha=0.02 \
    --mu=0.5 \
    --beta=1.0 \
    --norm=2 \
    --gamma=100 \
    --kappa=100 \
    --start=0 \
    --end=-1 \
    --model=sam_efficient \
    --model_type=vit_t \
