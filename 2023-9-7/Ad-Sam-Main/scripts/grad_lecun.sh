#!/bin/bash

CUDA_VISIBLE_DEVICES_LIST=(0)


for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python grad_null_text_inversion_edit.py \
    --save_root=output/lecun-box-Grad \
    --data_root=lecun_photo \
    --control_mask_dir=lecun_photo  \
    --caption_path=lecun_photo/blip2-caption.json\
    --inversion_dir=output/lecun-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa_000002_lecun_finetune.pth \
    --prompt_bs=140 \
    --eps=0.2 \
    --steps=10 \
    --alpha=0.02 \
    --mu=0.5 \
    --beta=1.0 \
    --norm=2 \
    --gamma=100 \
    --kappa=100 \
    --start=0 \
    --end=-1
    now=$(expr $now + $interval) 

done

wait