export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30007  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_adv@4~8 \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=4 \
--find_unused_params \
--numworkers=0 \
--eval \
