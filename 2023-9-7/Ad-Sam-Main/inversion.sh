#!/bin/bash

CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
#CUDA_VISIBLE_DEVICES_LIST=(0)
now=11187
interval=250

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python null_text_inversion.py \
    --save_root=output/sa_000001-Inversion \
    --data_root=sam-1b/sa_000001 \
    --control_mask_dir=sam-1b/sa_000001 \
    --caption_path=sam-1b/sa_000001-blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=${now} \
    --end=$((now + interval))\ &
    now=$(expr $now + $interval) 
done

wait
