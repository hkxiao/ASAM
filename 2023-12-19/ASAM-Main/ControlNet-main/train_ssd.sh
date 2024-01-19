export MODEL_DIR="../ckpt/SSD-1B"
export OUTPUT_DIR="train_output_ssd"
export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch diffusers/examples/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=sa_000000 \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --max_train_steps=30000 \
 --validation_image "./sa_1.png" "./sa_2.png" "./sa_3.png" \
 --validation_prompt "a stairway leading up to a building with ivy growing on it" "a white and red airplane sitting on a runway" "a group of people standing on a wooden pier" \
 --validation_steps=400 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --report_to="wandb" \
 --seed=42 \
 --controlnet_model_name_or_path=train_output_ssd/checkpoint-30000/controlnet \
 --hub_token=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