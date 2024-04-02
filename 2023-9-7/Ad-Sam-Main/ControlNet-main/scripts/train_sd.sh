export MODEL_DIR="/data/tanglv/xhk/ASAM/2023-9-7/Ad-Sam-Main/ckpt/stable-diffusion-v1-5"
export OUTPUT_DIR="train_output_sd_canny"

accelerate launch diffusers/examples/controlnet/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=sa_000000_canny \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "sa_000000_canny/conditioning_images/sa_1.png" "sa_000000_canny/conditioning_images/sa_2.png" "sa_000000_canny/conditioning_images/sa_3.png" "sa_000000_canny/conditioning_images/sa_4.png" \
 --validation_prompt "a stairway leading up to a building with ivy growing on it" "a white and red airplane sitting on a runway" "a group of people standing on a wooden pier" "a statue of a horse and a statue of a harry potter"\
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --seed=42 \
 --report_to=all \
 --controlnet_model_name_or_path=/data/tanglv/xhk/ASAM/2023-9-7/Ad-Sam-Main/ckpt/control_v11p_sd15_canny \
 --hub_token=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