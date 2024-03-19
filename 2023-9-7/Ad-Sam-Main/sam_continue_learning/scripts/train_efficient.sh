export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  efficient_sam_tuning.py \
--model-type vit_b \
--output_prefix efficient_sam_token-tuning_adv@4_dataset_sa000000adv_dice \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=2e-3 \
--train-datasets=dataset_sa000000adv_dice \
--valid-datasets=dataset_hrsod_val \
--slow_start \
--prompt_type box
--eval \
--restore-sam-model work_dirs/efficient_sam_token-tuning_adv@4t-efficient-vit_b-11186/epoch_9.pth \
--baseline \