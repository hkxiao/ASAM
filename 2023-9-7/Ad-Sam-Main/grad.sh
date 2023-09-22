CUDA_VISIBLE_DEVICES=0 python grad_null_text_inversion_edit.py --start=8850 --end=8900 &
CUDA_VISIBLE_DEVICES=1 python grad_null_text_inversion_edit.py --start=8900 --end=9000 &
CUDA_VISIBLE_DEVICES=2 python grad_null_text_inversion_edit.py --start=9850 --end=9900 &
CUDA_VISIBLE_DEVICES=3 python grad_null_text_inversion_edit.py --start=9900 --end=10000 &
CUDA_VISIBLE_DEVICES=4 python grad_null_text_inversion_edit.py --start=10850 --end=10900 &
CUDA_VISIBLE_DEVICES=5 python grad_null_text_inversion_edit.py --start=10900 --end=10950 &
CUDA_VISIBLE_DEVICES=6 python grad_null_text_inversion_edit.py --start=10950 --end=11000 &
