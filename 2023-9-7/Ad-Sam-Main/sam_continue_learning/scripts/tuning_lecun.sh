export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=1 --master_port=30013  main.py \
    --model-type vit_b \
    --output_prefix sam-baseline \
    --batch_size_train=1 \
    --batch_size_prompt=140 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --learning_rate=3e-2 \
    --train-datasets dataset_lecun \
    --valid-datasets dataset_lecun dataset_lecun_adv \
    --slow_start \
    --max_epoch_num 20 \
    --warmup_epoch 10 \
    --prompt_type point \
    --point_num 10 \
    --restore-model work_dirs/fine-tuning-4-dice-vit_b-11186/epoch_12.pth \
    --eval \
    --visualize2 \
    # --baseline
