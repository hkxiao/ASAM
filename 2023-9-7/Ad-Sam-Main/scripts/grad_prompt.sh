#!/bin/bash
#export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64/stubs/:/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/cudnn/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES_LIST=(0)
now=1
interval=5

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    # ''' 
    #     alpha: step_sieze; gamma: ce loss weight;  kappa: dice loss weight;
    # '''
    python grad_null_text_inversion_edit_prompt.py \
        --save_root=output/sa_000000-Grad \
        --data_root=sam-1b/sa_000000 \
        --control_mask_dir=sam-1b/sa_000000 \
        --caption_path=sam-1b/sa_000000-blip2-caption.json \
        --inversion_dir=output/sa_000000-Inversion/embeddings \
        --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
        --prompt_bs=4 \
        --eps=0.2 \
        --steps=10 \
        --alpha=0.01 \
        --mu=0.5 \
        --beta=0.5 \
        --norm=2 \
        --gamma=100 \
        --kappa=100 \
        --start=${now} \
        --end=$((now + interval)) \
        --attack_object=prompt   &
        now=$(expr $now + $interval) 
done

wait