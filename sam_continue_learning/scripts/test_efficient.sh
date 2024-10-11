export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  efficient_sam_tuning.py \
--model-type vit_t \
--output_prefix efficient_sam_token-tuning_adv@4t \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=3e-2 \
--train-datasets dataset_sa000000efficient \
--valid-datasets dataset_coco2017_val dataset_LVIS dataset_voc2012_val dataset_cityscapes_val  dataset_ade20k_val \
--slow_start \
--prompt_type box \
--eval \
--restore-model work_dirs/efficient_asam_token-tuning_no-data-augmentation@4-sa000000-11186-vit_t-1.0-1.0-0.03-10-20/epoch_19.pth


#--valid-datasets dataset_coco2017_val dataset_LVIS dataset_voc2012_val dataset_cityscapes_val  dataset_ade20k_val \
#--valid-datasets dataset_hrsod_val dataset_camo dataset_big_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_streets dataset_NDD20_ABOVE dataset_NDD20_BELOW dataset_TimberSeg dataset_ppdls dataset_ishape_antenna dataset_visor_val dataset_ibd_val dataset_woodscape dataset_PIDRAY dataset_DRAM_test dataset_TrashCan_val dataset_gtea_train dataset_Kvasir_sessile dataset_Kvasir_SEG dataset_CVC_ClinicDB \
