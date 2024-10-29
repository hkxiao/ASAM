#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tutorial_train.py --batchsize=4 --gpus 8 --steps 30000 --dataset ../data/sam-1b \
    --json_path ../data/sam-1b/sa_000000-sa_000004-controlnet-feat-train.json \
    --resume=models/control_sd15_ini_feat_control_24_channel.ckpt \
    --config=models/cldm_v15_feat_control_24_channel.yaml