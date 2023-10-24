export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=1 --master_port=30006  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_PGD@4 \
--find_unused_params \
--eval \
--restore-model work_dirs/sam_token-tuning_b_PGD@4/epoch_18.pth 
