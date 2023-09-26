export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30002  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_decoder-tuning_b_origin@4 \
--batch_size_train=8 \
--batch_size_prompt=4 \
--find_unused_params
