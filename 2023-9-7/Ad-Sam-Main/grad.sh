#!/bin/bash

#CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6)
#CUDA_VISIBLE_DEVICES_LIST=(1)
CUDA_VISIBLE_DEVICES_LIST=(6 7)
now=1
interval=6000

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python grad_null_text_inversion_edit.py \
    --save_root=output/sa_000000@4-Grad \
    --data_root=/data/tanglv/data/sam-1b/sa_000000 \
    --control_mask_dir=/data/tanglv/data/sam-1b/sa_000000/four_mask \
    --caption_path=/data/tanglv/data/sam-1b/sa_000000-blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000000@4.pth \
    --inversion_dir=/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/output/sa_000000@4-Inversion/embeddings \
    --sam_batch=4 \
    --model_type=vit_b \
    --guidance_scale=9.0 \
    --ddim_steps=20 \
    --eps=0.2 \
    --steps=10 \
    --alpha=0.02 \
    --mu=0.5 \
    --beta=0.1 \
    --norm=2 \
    --gamma=10 \
    --start=${now} \
    --end=$((now + interval)) &
    now=$(expr $now + $interval) 
done

wait
