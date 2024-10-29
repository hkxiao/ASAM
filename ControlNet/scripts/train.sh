#! /bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
python tutorial_train.py --batchsize=2 --gpus 1 --epoch 10 --dataset ../data/sam-1b/sa_000000 \
    --json_path ../data/sam-1b/sa_000000-controlnet-train-mini.json \
    --resume=models/control_sd15_ini_feat_control.ckpt \
    --config=models/cldm_v15_feat_control.yaml