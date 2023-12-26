#!/bin/bash

CUDA_VISIBLE_DEVICES_LIST=(0)
now=6000
interval=1

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python grad_null_text_inversion_edit.py \
    --save_root=output/sa_000001-Grad \
    --data_root=sam-1b/sa_000001 \
    --control_mask_dir=sam-1b/sa_000001 \
    --caption_path=sam-1b/sa_000001-blip2-caption.json \
    --inversion_dir=output/sa_000001-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
    --sam_batch=140 \
    --eps=0.2 \
    --steps=10 \
    --alpha=0.02 \
    --mu=0.5 \
    --beta=1.0 \
    --norm=2 \
    --gamma=100 \
    --kappa=100 \
    --start=${now} \
    --end=$((now + interval)) 
    now=$(expr $now + $interval) 

done

wait