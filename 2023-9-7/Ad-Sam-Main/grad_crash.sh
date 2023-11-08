#!/bin/bash

start_program() {
    export CUDA_VISIBLE_DEVICES=$1
    python $2 \
    --save_root=output/sa_000000-Grad \
    --data_root=sam-1b/sa_000000 \
    --control_mask_dir=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
    --sam_batch=140 \
    --eps=0.2 \
    --steps=10 \
    --alpha=0.01 \
    --mu=0.5 \
    --beta=1.0 \
    --norm=2 \
    --gamma=100 \
    --kappa=100 \
    --start=$3 \
    --end=$4
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
start=(1 500 1000 1500 2000 2500 3000 3500)
end=(500 1000 1500 2000 2500 3000 3500 4000)

# CUDA_VISIBLE_DEVICES_LIST=(6)
# start=(1)
# end=(1)

PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} grad_null_text_inversion_edit.py ${start[i]} ${end[i]} &
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
            
            # 获取进程的退出状态
            start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} utils/grad_crash_aid.py ${start[i]} ${end[i]}

            STATUS[i]=$?
            echo "进程 $process_id 的退出状态为 ${STATUS[i]}"

            if [ ${STATUS[i]} -ne 0 ]; then
                start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} grad_null_text_inversion_edit.py ${start[i]} ${end[i]} &
                PID_LIST[i]=$!
                echo "进程 ${PID_LIST[i]} 重新执行"
                finish=false
            fi
        fi
    done
    if [ $finish == true ]; then
        break
    fi
    sleep 1
done