export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=7 --master_port=30001  train_samsubset.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/hq_sam_b_origin 
