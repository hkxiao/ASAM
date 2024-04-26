#!/bin/bash

#CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES_LIST=(0)
now=1543967
interval=1500

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python null_text_inversion_lecun.py \
    --save_root=output/lecun-Inversion \
    --data_root=/data/tanglv/xhk/segment-anything/lecun \
    --control_mask_dir=/data/tanglv/xhk/segment-anything/lecun \
    --caption_path=/data/tanglv/xhk/segment-anything/lecun/blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa_000002_lecun_finetune.pth \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=${now} \
    --end=$((now + interval))\ 
    now=$(expr $now + $interval) 
done

wait
