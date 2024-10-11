# Method nima-vgg16-ava hyperiqa musiq-ava musiq-koniq tres
export CUDA_VISIBLE_DEVICES=5
python test_quality.py \
--img_path /data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/sam_continue_learning/train/work_dirs/sam_token-tuning_pgd_tuning@4/adv_examples \
--metric tres