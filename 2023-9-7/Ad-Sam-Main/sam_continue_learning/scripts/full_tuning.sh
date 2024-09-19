export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  sam_tuning.py \
    --model-type vit_b \
    --output_prefix sam-full_tuning \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --learning_rate=1e-6 \
    --train-datasets dataset_sa000000_SSD_adv_imgs_prompt \
    --valid-datasets dataset_hrsod_val \
    --slow_start \
    --max_epoch_num 12 \
    --warmup_epoch 8 \

python /data/tanglv/xhk/occupy/a40.py --gpus 0 1 2 3 4 5 6 7
#--learning_rate=3e-6 \