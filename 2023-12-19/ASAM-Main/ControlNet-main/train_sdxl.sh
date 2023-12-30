export MODEL_DIR="../ckpt/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="train_output_sdxl"

accelerate launch diffusers/examples/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --hub_token=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