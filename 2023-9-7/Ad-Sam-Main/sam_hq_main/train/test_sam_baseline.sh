export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python -m torch.distributed.launch --nproc_per_node=6 --master_port=31008  test_sam_baseline.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b \
--output work_dirs/sam_b—baseline \
--eval \
--visualize



