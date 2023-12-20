<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30004  main.py \
--model-type vit_l \
--output_prefix sam_token-tuning_adv@4 \
=======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --master_port=30004  pgd_tuning.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_token-tuning_pgd_tuning@4 \
>>>>>>> 162f1bff4a5daafbf870d7f43445bd1dbf3370b2
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
<<<<<<< HEAD
--learning_rate=5e-3 \
--train-datasets=dataset_sa000000adv_dice \
--valid-datasets=dataset_hrsod_val \
--max_epoch_num=35
=======
--learning_rate=1e-3 \
--train-datasets=dataset_sam_subset_ori \
--valid-datasets=dataset_hrsod_val \
>>>>>>> 162f1bff4a5daafbf870d7f43445bd1dbf3370b2
