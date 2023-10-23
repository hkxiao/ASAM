export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run --nproc_per_node=2 --master_port=30005  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_adv@8~16 \
--find_unused_params \
--eval \
--restore-model work_dirs/sam_token-tuning_b_adv@8~16/epoch_19.pth 
