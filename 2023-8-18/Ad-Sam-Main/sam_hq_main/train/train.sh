export CUDA_VISIBLE_DEVICES=0,1,3,4
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30000  train.py --checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b 
