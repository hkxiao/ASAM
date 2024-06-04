#!/bin/bash

CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
now=1
interval=5

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python grad_null_text_inversion_edit.py \
    --save_root=vis_demo/sa_000000-Grad \
    --data_root=sam-1b/sa_000000 \
    --control_mask_dir=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --inversion_dir=output/sa_000000-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
    --prompt_bs=140 \
    --eps=0.2 \
    --steps=2 \
    --alpha=0.05 \
    --mu=0.5 \
    --beta=0.3 \
    --norm=2 \
    --gamma=500 \
    --kappa=500 \
    --start=${now} \
    --end=$((now + interval)) &
    now=$(expr $now + $interval) 
done

wait