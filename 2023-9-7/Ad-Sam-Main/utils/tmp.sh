#! /bin/bash

python sam_control_signal.py

cd ../ControlNet-main
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python tutorial_train.py --gpus 6 --dataset ../sam-subset-11187


