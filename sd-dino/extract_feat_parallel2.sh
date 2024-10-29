#!/bin/bash
#export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64/stubs/:/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/cudnn/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)


CUDA_VISIBLE_DEVICES_LIST=(2 3)
#CUDA_VISIBLE_DEVICES_LIST=
now=1
interval=5593

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    # ''' 
    #     alpha: step_sieze; gamma: ce loss weight;  kappa: dice loss weight;
    # '''
    python extract_sd_raw+dino.py --img_dir=../data/sam-1b/sa_000002 \
        --caption_path=../data/sam-1b/sa_000002-blip2-caption.json \
        --output_dir output1 \
        --start=${now} \
        --end=$((now + interval))&

    now=$(expr $now + $interval) 
done
