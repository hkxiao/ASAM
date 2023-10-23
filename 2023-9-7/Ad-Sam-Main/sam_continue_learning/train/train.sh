export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30004  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_adv@8~16 \
--batch_size_train=8 \
--batch_size_prompt=8 \
--batch_size_prompt_start=8 \
--find_unused_params \
--numworkers=0
