#! /bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
python tutorial_train.py --batchsize=4 --gpus 4 --epoch 10 --dataset /data/tanglv/xhk/segment-anything/lecun \
--json_path /data/tanglv/xhk/segment-anything/lecun/contronet_config.json \
--resume=models/control_sd15_ini.ckpt \
--resume_controlnet=../ckpt/control_v11p_sd15_mask_sa000002.pth