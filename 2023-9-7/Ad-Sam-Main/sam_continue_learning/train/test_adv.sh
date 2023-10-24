export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30007  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_adv@8 \
--batch_size_train=8 \
--batch_size_prompt=8 \
--batch_size_prompt_start=0 \
--find_unused_params \
--restore-model work_dirs/sam_token-tuning_b_adv@8/epoch_19.pth \
--numworkers=0 \
--eval 