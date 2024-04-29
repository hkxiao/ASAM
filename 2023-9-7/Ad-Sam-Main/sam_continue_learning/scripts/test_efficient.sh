export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --nproc_per_node=4 --master_port=30011  efficient_sam_tuning.py \
--model-type vit_t \
--output_prefix efficient_sam_token-tuning_adv@4t \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=2e-3 \
--train-datasets dataset_sa000000efficient \
--valid-datasets dataset_ovis_train \
--slow_start \
--prompt_type box \
--restore-model pretrained_checkpoint/efficient_sam_vitt.pt \
--eval


#--valid-datasets dataset_coco2017_val dataset_LVIS dataset_voc2012_val dataset_cityscapes_val  dataset_ade20k_val \
#--valid-datasets dataset_hrsod_val dataset_camo dataset_big_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos dataset_streets dataset_NDD20_ABOVE dataset_NDD20_BELOW dataset_TimberSeg dataset_ppdls dataset_ishape_antenna dataset_visor_val dataset_ibd_val dataset_woodscape dataset_PIDRAY dataset_DRAM_test dataset_TrashCan_val dataset_gtea_train dataset_Kvasir_sessile dataset_Kvasir_SEG dataset_CVC_ClinicDB \
