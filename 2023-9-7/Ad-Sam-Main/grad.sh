CUDA_VISIBLE_DEVICES=2 python grad_null_text_inversion_edit.py --start=1 --end=5 &
CUDA_VISIBLE_DEVICES=3 python grad_null_text_inversion_edit.py --start=6 --end=10 &
CUDA_VISIBLE_DEVICES=4 python grad_null_text_inversion_edit.py --start=11 --end=15 & 
CUDA_VISIBLE_DEVICES=5 python grad_null_text_inversion_edit.py --start=16 --end=20 &
CUDA_VISIBLE_DEVICES=6 python grad_null_text_inversion_edit.py --start=21 --end=25&
CUDA_VISIBLE_DEVICES=7 python grad_null_text_inversion_edit.py --start=26 --end=30&