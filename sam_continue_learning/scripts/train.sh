export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run --nproc_per_node=2 --master_port=30011  main.py \
    --model-type vit_b \
    --output_prefix sam-token_tuning_adv_prompt_v4 \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --learning_rate=3e-2 \
    --train-datasets dataset_sa000000_adv_imgs_prompt \
    --valid-datasets dataset_hrsod_val \
    --slow_start \
    --max_epoch_num 20 \
    --warmup_epoch 10 \
    --train_prompt_types points boxes \
    --eval_prompt_type boxes \
    --data_augmentation False \
    --eval_multimask_output False \
    --eval_stability True \
    --adv_prompt_training True \
    --eval_record True \
    --visualize \

python /data/tanglv/xhk/occupy/a40.py --gpus 0 1 2 3 4 5 6 7
#--learning_rate=3e-6 \