#!/bin/bash

#CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
#CUDA_VISIBLE_DEVICES_LIST=(0)
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
now=0
interval=1400
for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python inversion.py \
    --save_root=work_dirs/sa_000000-Direction-Inversion \
    --data_root=data/sam-1b/sa_000000 \
    --control_mask_dir=data/sam-1b/sa_000000 \
    --control_feat_dir=sd-dino/work_dirs/sa_000000 \
    --caption_path=data/sam-1b/sa_000000-blip2-caption.json \
    --mask_controlnet_path=pretrained/control_v11p_sd15_mask_sa000001.pth \
    --feat_controlnet_path=pretrained/control_v11p_sd15_feat_sa000000~4.pth \
    --mask_conditioning_channels 3 \
    --feat_conditioning_channels 24 \
    --mask_control_scale 0.5 \
    --feat_control_scale 0.5 \
    --sd_path=runwayml/stable-diffusion-v1-5 \
    --inversion_type direct_inversion \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=${now} \
    --end=$((now + interval))\ &
    now=$(expr $now + $interval) 
done

wait