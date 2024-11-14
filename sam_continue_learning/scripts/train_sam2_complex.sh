#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=8 --master_port=30001  main.py \
    --model-type vit_t \
    --output_prefix sam2.0-full_tuning-hq \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --train_prompt_type boxes points \
    --eval_prompt_type boxes \
    --train-datasets dataset_complex \
    --valid-datasets dataset_coift_val dataset_hrsod_val dataset_thin_val dataset_dis_val \
    --stable_iter 10 \
    --boxes_noise_scale 0.2 \
    --eval_multimask_output False \
    --mask_id 2 \
    --eval_record True \
    --eval_stability True \
    --eval_stability_with gt \
    --serial_prompt True \
    --serial_prompt_size 100 \
    --eval_img_num -1 \
    --sam-type sam2 \
    --model-config "configs/sam2/sam2_hiera_t.yaml" \
    --two_stage False \
    --tuning_manner full_weights \
    --slow_start \
    --max_epoch_num 14 \
    --warmup_epoch 8 \
    --alpha 1.0 \
    --beta 1.0 \
    --learning_rate=1e-7 \
    --train_vis False

--restore-sam-model /remote-home/tanglv/ASAM/sam_continue_learning/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000-large-scale_finetune.yaml/checkpoints/checkpoint.pt \
--restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000_finetune.yaml/checkpoints/checkpoint.pt \
--restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Ori-sa000000_finetune.yaml/checkpoints/checkpoint.pt
# --restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000_finetune.yaml/checkpoints/checkpoint.pt \
# --restore-sam-model-keys model \