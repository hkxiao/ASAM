#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=8 --master_port=30001  main.py \
    --model-type vit_t \
    --output_prefix sam2.1-full_tuning-gt_boxes \
    --only_output_prefix False \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --train_prompt_type boxes points \
    --eval_prompt_type points \
    --train-datasets dataset_sa000000_direct_inversion_adv_augmentation_img_gt_prompt_sam2 \
    --valid-dataset dataset_visor_val \
    --stable_iter 10 \
    --boxes_noise_scale 0.2 \
    --eval_multimask_output False \
    --mask_id 2 \
    --eval_record True \
    --eval_stability False \
    --eval_stability_with gt \
    --serial_prompt True \
    --serial_prompt_size 100 \
    --eval_img_num -1 \
    --sam-type sam2 \
    --model-config "configs/sam2/sam2_hiera_t.yaml" \
    --two_stage False \
    --tuning_manner full_weights \
    --slow_start \
    --max_epoch_num 12 \
    --warmup_epoch 8 \
    --alpha 1.0 \
    --beta 1.0 \
    --learning_rate=1e-7 \
    --restore-sam-model work_dirs/sam2.0-full_tuning-hq-dis-thin-fss-duts-duts_te-ecssd-msra-vit_t--1/epoch_13.pth \
    --eval

# dataset_Kvasir_SEG dataset_CVC_ClinicDB dataset_ppdls dataset_visor_val dataset_ibd_val dataset_PIDRAY dataset_DRAM_test
# --restore-sam-model /remote-home/tanglv/ASAM/sam_continue_learning/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000-large-scale_finetune.yaml/checkpoints/checkpoint.pt \
# --restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000_finetune.yaml/checkpoints/checkpoint.pt \
# --restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Ori-sa000000_finetune.yaml/checkpoints/checkpoint.pt
# --restore-sam-model sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_tiny_Adv-sa000000_finetune.yaml/checkpoints/checkpoint.pt \
# --restore-sam-model-keys model \