export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30011  main.py \
--model-type vit_b \
--output_prefix sam_token-tuning_adv_15000@4 \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=1e-3 \
--train-datasets=dataset_sa000001adv_dice \
--valid-datasets=dataset_hrsod_val \
--slow_start \
--train_img_num=4000 \
--restore-model=work_dirs/diceloss_sam_iou_masktoken-tuning_b_adv@4/epoch_9.pth

export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30011  main.py \
--model-type vit_b \
--output_prefix work_dirs/diceloss_sam_iou_masktoken-tuning_b_adv@4 \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--restore-model work_dirs/sam_token-tuning_adv_150004-dice-vit_b-4000/epoch_9.pth \
--eval \   
--prompt_type box \
--train-datasets dataset_sa000001adv_dice \
--valid-datasets dataset_voc2012_val dataset_cityscapes_val dataset_coco2017_val dataset_LVIS dataset_ade20k_val \
--visualize \