export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30004  sam_tuning.py \
--model-type vit_b \
--output_prefix sam_entire-tuning_adv@4 \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=1e-4 \
--train-datasets=dataset_sa000000adv_dice \
--valid-datasets=dataset_voc2012_val dataset_cityscapes_val dataset_coco2017_val  dataset_LVIS  dataset_ade20k_val \
--max_epoch_num=12 \
--slow_start
#--train-datasets=dataset_sa000000adv_dice \