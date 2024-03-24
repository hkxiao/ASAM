# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run --nproc_per_node=4 --master_port=30005 pgd_tuning.py \
--model-type vit_b \
--output_prefix sam-baseline \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--eval \
--prompt_type box \
--train-datasets dataset_sa000000 \
--valid-datasets dataset_hrsod_val dataset_camo dataset_DRAM_test dataset_PIDRAY dataset_ibd_val dataset_coco2017_val \
--restore-model work_dirs/sam_token-tuning_pgd_tuning@4/epoch_9.pth \
--visualize
# --baseline \
# --valid-datasets dataset_hrsod_val dataset_camo dataset_DRAM_test dataset_PIDRAY dataset_ibd_val dataset_coco2017_val \
# --baseline
# --visualize \
#--baseline
#--valid-datasets dataset_camo dataset_big_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_LVIS \
#--valid-datasets dataset_hrsod_val dataset_ade20k_val dataset_voc2012_val dataset_cityscapes_val dataset_coco2017_val dataset_camo dataset_big_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_LVIS \
#--valid-datasets dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_LVIS \
#--baseline
#--restore-model work_dirs/sam_token-tuning_pgd_tuning@4/epoch_9.pth 
