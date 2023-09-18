CUDA_VISIBLE_DEVICES=1 python null_text_inversion_batch_11187_prompts.py --start=1 --end=1000 &
CUDA_VISIBLE_DEVICES=2 python null_text_inversion_batch_11187_prompts.py --start=1000 --end=2000 &
CUDA_VISIBLE_DEVICES=3 python null_text_inversion_batch_11187_prompts.py --start=2000 --end=3000 &
CUDA_VISIBLE_DEVICES=4 python null_text_inversion_batch_11187_prompts.py --start=3000 --end=4000 &
CUDA_VISIBLE_DEVICES=5 python null_text_inversion_batch_11187_prompts.py --start=4000 --end=5000 &
CUDA_VISIBLE_DEVICES=6 python null_text_inversion_batch_11187_prompts.py --start=5000 --end=6000 &
CUDA_VISIBLE_DEVICES=7 python null_text_inversion_batch_11187_prompts.py --start=6000 --end=7000