export MODEL_DIR="../ckpt/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="train_output_sdxl"

accelerate launch diffusers/examples/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=../sam-1b/sa_000000 \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "../sam-1b/sa_000000/sa_1.png" "../sam-1b/sa_000000/sa_2.png" "../sam-1b/sa_000000/sa_3.png" "../sam-1b/sa_000000/sa_4.png" \
 --validation_prompt "a stairway leading up to a building with ivy growing on it" "a white and red airplane sitting on a runway" "a group of people standing on a wooden pier" "a statue of a horse and a statue of a harry potter"\
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --seed=42 \
 --report_to=all \
 --hub_token=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