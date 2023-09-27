export CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.launch --nproc_per_node=1 --master_port=31011  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_b \
--eval \
--base


