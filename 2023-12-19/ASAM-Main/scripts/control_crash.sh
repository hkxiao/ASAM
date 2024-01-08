#!/bin/bash

start_program() {
    export CUDA_VISIBLE_DEVICES=$1
    python $2 \
    --save_root=output/sa_000000-Control \
    --data_root=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --inversion_dir=output/Inversion/SD-7.5-50-INV-5/ \
    --grad_dir=output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-160-ADV-0.4-4-0.1-0.5-100.0-100.0-1.0-2/ \
    --control_mask_dir=sam-1b/sa_000000 \
    --controlnet_path=ControlNet-main/train_output_sdxl/checkpoint-15000/controlnet \
    --ddim_steps=1 \
    --control_scale=0.0 \
    --start=$3 \
    --end=$4
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
start=(1 500 1000 1500 2000 2500 3000 3500)
end=(500 1000 1500 2000 2500 3000 3500 4000)

PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} control_edit.py ${start[i]} ${end[i]} &
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