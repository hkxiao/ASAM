export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=6 --master_port=30001  main.py \
--model-type vit_b \
--output_prefix asam-controlnet@4-control-vit_b-11186 \
--find_unused_params \
--numworkers=0 \
--restore-model work_dirs/asam-controlnet@4-control-vit_b-11186/epoch_9.pth \
--eval \
--prompt_type box \
--train-datasets=dataset_sa000000_control \
--valid-datasets dataset_camo dataset_big_val dataset_hrsod_val dataset_voc2012_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_LVIS \
# --visualize 