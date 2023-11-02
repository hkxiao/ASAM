#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tutorial_train.py --batchsize=8 --gpus 8 --epoch 30 --dataset /data/tanglv/data/sam-1b/sa_000000  --json_path /data/tanglv/data/sam-1b/sa_000000@4-contrtolnet-train.json \
--resume=/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/ControlNet-main/models/control_sd15_ini.ckpt