export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  efficient_sam_tuning.py \
    --model-type vit_t \
    --output_prefix efficient_asam_token-tuning_no-data-augmentation@4 \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --learning_rate=3e-2 \
    --max_epoch_num=20 \
    --slow_start \
    --warmup_epoch=10 \
    --train-datasets dataset_sa000000 \
    --valid-datasets dataset_hrsod_val dataset_coco2017_val \
    --prompt_type box \
    --alpha 1.0 \
    --beta 1.0 
