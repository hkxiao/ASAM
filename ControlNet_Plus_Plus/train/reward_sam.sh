export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="../ckpt/stable-diffusion-v1-5"
export CONTROLNET_DIR="../ckpt/control_v11p_sd15_mask_sa000002.pth"

export CONTROLNET_DIR="/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/ControlNet-main/train_output_ssd/checkpoint-10500/controlnet"
export MODEL_DIR="../ckpt/stable-diffusion-xl-base-1.0"

#export CONTROLNET_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM/checkpoint-4500/controlnet"
#export CONTROLNET_DIR="work_dirs/finetune/Captioned_ADE20K/ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16/checkpoint-5000/controlnet"
export OUTPUT_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_lr1e-6"
#export OUTPUT_DIR="work_dirs/reward_model/SAM-1B/tmp"

# Download our fine-tuned weights
# You can also train a new one with command `bash train/finetune_ade20k.sh`

# reward fine-tuning
accelerate launch --config_file "train/config.yml" \
    --main_process_port=23158 train/reward_control.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --controlnet_model_name_or_path=$CONTROLNET_DIR \
    --output_dir=$OUTPUT_DIR \
    --task_name="interactive_segmentation" \
    --dataset_name="./sa000000" \
    --image_column="image" \
    --caption_column="text" \
    --conditioning_image_column="conditioning_image" \
    --label_column="label_dir" \
    --cache_dir="data/huggingface_datasets" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=16 \
    --learning_rate=1e-6 \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --dataloader_num_workers=16 \
    --max_train_steps=10000 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=200 \
    --checkpointing_steps=1 \
    --grad_scale=0.5 \
    --use_ema \
    --validation_steps=10 \
    --timestep_sampling_start=0 \
    --timestep_sampling_end=1000 \
    --min_timestep_rewarding=0 \
    --max_timestep_rewarding=200 \
    --report_to "wandb" \
#--resume_from_checkpoint "work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM/checkpoint-6500"
    
#python /data/tanglv/xhk/occupy/a40.py --gpus 0 1 2 3 4 5 6 7


##wandb api key:  a097761ce0356756721ed6f44fe311b555e3d0cd