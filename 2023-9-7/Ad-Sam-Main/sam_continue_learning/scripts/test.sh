export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CUDA_VISIBLE_DEVICES=3,7,0
python -m torch.distributed.run --nproc_per_node=3 --master_port=30005  main.py \
    --model-type vit_b \
    --output_prefix sam-baseline \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --eval \
    --train_prompt_type boxes_points \
    --eval_prompt_type points \
    --train-datasets dataset_sa000000 \
    --valid-datasets dataset_hrsod_val \
    --mask_id 2 \
    --stable_iter 10 \
    --boxes_noise_scale 0.2 \
    --restore-model work_dirs/asam-now/epoch_11*.pth \
    --eval

--eval_stability \