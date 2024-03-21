export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  main.py \
--model-type vit_b \
--output_prefix fine-tuning \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=2e-1 \
--train-datasets=dataset_sa000000adv_dice \
--valid-datasets dataset_coco2017_val dataset_hrsod_val \
--slow_start \
--max_epoch_num 40 \
--warmup_epoch 30 \
--restore-model=work_dirs/sam_token-tuning_adv_point-prompt@4-dice-vit_b-11186/epoch_34.pth \