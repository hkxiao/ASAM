export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=31007  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output sam_token-tuning_b_origin@4 \
--eval \
--restore-model work_dirs/sam_token-tuning_b_origin@4/epoch_19.pth 


