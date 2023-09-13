export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=31007  test.py --checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b --eval --restore-model /data/tanglv/Ad-SAM/2023-8-18/Ad-Sam-Main/sam_hq_main/train/work_dirs/hq_sam_b/epoch_8*.pth
