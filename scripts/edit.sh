subset="sa_000000"
start=390
end=400

python adv_edit.py \
    --save_root=work_dirs/${subset}-Grad \
    --data_root=data/sam-1b/${subset} \
    --control_mask_dir=data/sam-1b/${subset} \
    --control_feat_dir=sd-dino/work_dirs/${subset} \
    --caption_path=data/sam-1b/${subset}-blip2-caption.json \
    --mask_conditioning_channels 3 \
    --feat_conditioning_channels 24 \
    --mask_control_scale 0.5 \
    --feat_control_scale 0.5 \
    --inversion_dir=work_dirs/${subset}-Direction-Inversion/embeddings \
    --inversion_type direct_inversion \
    --mask_controlnet_path=pretrained/control_v11p_sd15_mask_sa000001.pth \
    --feat_controlnet_path=pretrained/control_v11p_sd15_feat_sa000000~4.pth \
    --sd_path=runwayml/stable-diffusion-v1-5 \
    --prompt_bs=4 \
    --eps=0.2 \
    --eps_boxes=20.0 \
    --eps_points=20.0 \
    --ddim_steps=50 \
    --steps=10 \
    --alpha=0.01 \
    --alpha_boxes=4 \
    --alpha_points=4 \
    --boxes_noise_scale=0.1 \
    --points_noise_scale=0.1 \
    --mu=0.5 \
    --beta=0.5 \
    --norm=2 \
    --gamma=100 \
    --kappa=100 \
    --attack_object image points boxes \
    --prompt_type  points boxes \
    --embedding_sup True \
    --embedding_mse_weight 0.5 \
    --start=${start} \
    --end=${end} \
    --model 'sam2' \
    --model_type 'vit_t' \
    --model_config sam2_hiera_t.yaml \
    $5