#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tutorial_train.py --gpus 8 --dataset ../../../data/sod_data/DUTS-TR
