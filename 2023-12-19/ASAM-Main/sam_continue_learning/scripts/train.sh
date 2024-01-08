export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=7 --master_port=30004  main.py \
--model-type vit_b \
--output work_dirs/asam-controlnet@4 \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=1e-3 \
--train-datasets=dataset_sa000000_control \
--valid-datasets=dataset_hrsod_val \
