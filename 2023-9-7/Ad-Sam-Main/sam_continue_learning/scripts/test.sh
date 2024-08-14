#export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.run --nproc_per_node=8 --master_port=30009  main.py \
    --model-type vit_h \
    --output_prefix sam-baseline \
    --batch_size_train=4 \
    --batch_size_prompt=4 \
    --batch_size_prompt_start=0 \
    --find_unused_params \
    --numworkers=0 \
    --train_prompt_type boxes points \
    --eval_prompt_type boxes \
    --train-datasets dataset_sa000000 \
    --valid-datasets dataset_sa000138_controlnet dataset_sa000138_controlnet_plus_no_reward_1 dataset_sa000138_controlnet_plus_10 dataset_sa000138_controlnet_plus_500 dataset_sa000138_controlnet_plus_4500 dataset_sa000138_controlnet_plus_7000 \
    --stable_iter 10 \
    --boxes_noise_scale 0.2 \
    --restore-model work_dirs/sam-token_tuning_adv_prompt_v4-sa000000_adv_imgs_prompt-vit_b-11186/epoch_19.pth \
    --eval_multimask_output False \
    --mask_id 2 \
    --eval \
    --eval_stability False \
    --serial_prompt True \
    --serial_prompt_size 500 \
    --baseline \
    --eval_img_num 500 
    #--visualize


dataset_cityscapes_val dataset_ade20k_val dataset_voc2012_val dataset_LVIS
# dataset_hrsod_val dataset_cityscapes_val dataset_ade20k_val dataset_voc2012_val dataset_coco2017_val dataset_LVIS