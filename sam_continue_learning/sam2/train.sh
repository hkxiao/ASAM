# export path_to_img_folder='/remote-home/tanglv/ASAM/data/sam-1b'
# export path_to_gt_folder='/remote-home/tanglv/ASAM/data/sam-1b'
# python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
#     --use-cluster 0 \
#     --num-gpus 8


python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000-large-scale_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8