#! /bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python tutorial_train.py --gpus 6 --dataset ../sa_000001 --json_path sam-1b-controlnet-train_1.json \
--resume=lightning_logs/version_25/checkpoints/epoch=56-step=19949.ckpt