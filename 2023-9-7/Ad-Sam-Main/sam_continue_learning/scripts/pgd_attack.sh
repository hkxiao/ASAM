export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30008  pgd_tuning.py \
--model-type vit_b \
--output_prefix FSGM \
--find_unused_params \
--numworkers=0 \
--learning_rate=1e-4 \
--train-datasets=dataset_cityscapes_val \
--only_attack \
--valid-datasets=dataset_voc2012_val \
--max_epoch_num=1 \
--slow_start