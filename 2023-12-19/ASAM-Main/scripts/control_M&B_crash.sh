#!/bin/bash

start_program() {
    export CUDA_VISIBLE_DEVICES=$1
    python $2 \
    --save_root=output/sa_000000-Control \
    --data_root=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --inversion_dir=output/Inversion/SSD-7.5-50-INV-5/ \
    --grad_dir=output/sa_000000-Grad/skip-ablation-01-mi-SSD-7.5-50-SAM-sam-vit_b-160-ADV-0.2-2-0.1-0.5-32.0-32.0-1.0-2/ \
    --control_mask_dir=sam-1b/sa_000000 \
    --mask_controlnet_path=ControlNet-main/train_output_ssd/checkpoint-10500/controlnet \
    --boundary_controlnet_path=ControlNet-main/train_output_ssd_canny/checkpoint-30000/controlnet\
    --SD_path=ckpt/SSD-1B \
    --ddim_steps=50 \
    --mask_control_scale=0.0 \
    --boundary_control_scale=1.0 \
    --start=$3 \
    --end=$4 \
    --random_latent
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES_LIST=(0)
start=(1)
end=(1)




PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} mask\&boundary_control_edit.py ${start[i]} ${end[i]} &
    PID_LIST+=($!)
    STATUS+=(-1)
done

# check crash
while true
do
    finish=true
    for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
    do
        process_id=${PID_LIST[i]}
        # 检查进程是否在执行
        if [ ${STATUS[i]} -eq 0 ]; then
            echo "进程 $process_id 执行完成"
            continue
        elif ps -p $process_id > /dev/null; then
            echo "进程 $process_id 正在执行"
            finish=false
        else
            echo "进程 $process_id 未在执行"
        fi
    done
    if [ $finish == true ]; then
        break
    fi
    sleep 5
done