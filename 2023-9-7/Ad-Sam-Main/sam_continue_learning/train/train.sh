export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=30003  train_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_b_adv@16 \
--batch_size_train=8 \
--batch_size_prompt=16 \
--find_unused_params \
--restore-model=/data/tanglv/xhk/Ad-Sam-Main/sam_continue_learning/train/work_dirs/sam_token-tuning_b_adv@16/epoch_8.pth\
