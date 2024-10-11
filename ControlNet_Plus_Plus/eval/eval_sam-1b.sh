# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"  # Eval ControlNet
# export CONTROLNET_DIR="checkpoints/ade20k_reward-model-FCN-R101-d8/checkpoint-5000/controlnet"  # Eval our ControlNet++
# export CONTROLNET_DIR="../ckpt/control_v11p_sd15_mask_sa000001.pth"
#export CONTROLNET_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM/checkpoint-500/controlnet"
#export CONTROLNET_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_restart/checkpoint-10/controlnet"
#export CONTROLNET_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_restart/checkpoint-20/controlnet"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CONTROLNET_DIR="work_dirs/reward_model/SAM-1B/reward_ft_controlnet_sd15_interactive_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_EfficientSAM_restart/checkpoint-180/controlnet"
# export CONTROLNET_DIR="/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/ControlNet-main/train_output_sdxl/checkpoint-15000/controlnet"
export CONTROLNET_DIR="/data/tanglv/xhk/ASAM/2023-12-19/ASAM-Main/ControlNet-main/train_output_ssd/checkpoint-10500/controlnet"
export CONTROLNET_DIR="../ControlNetSDXL/train_output_sdxl3/checkpoint-2000/controlnet"
# export SD_PATH="../ckpt/stable-diffusion-xl-base-1.0"
export SD_PATH="../ckpt/SSD-1B"
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=6
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20
# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=23457 --num_processes=$NUM_GPUS eval/eval.py --task_name='interactive_segmentation' --dataset_name='./sa000000' --dataset_split='validation' --condition_column='conditioning_image' --prompt_column='text' --label_column='label_dir' --model controlnet-sdxl --model_path=${CONTROLNET_DIR} --sd_path=${SD_PATH} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/eval_dirs/sa000000/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"


# Evaluation with mmseg api
# mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
#     --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
#     --gpus 8 \
#     --launcher pytorch \
#     --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
#                   work_dir="${DATA_DIR}"
