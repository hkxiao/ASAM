export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --master_port=31008  test_sam.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/hq_sam_b_train-mask_4-ade20k \
--eval \
--restore-model /data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/sam_hq_main/train/work_dirs/hq_sam_b_demo/epoch_5.pth \
--visualize \


