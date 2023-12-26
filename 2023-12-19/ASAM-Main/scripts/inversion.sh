#!/bin/bash

CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
#CUDA_VISIBLE_DEVICES_LIST=(0)
now=1
interval=250

for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python null_text_inversion.py \
    --save_root=output/sa_000000-Inversion \
    --data_root=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=${now} \
    --end=$((now + interval))\ &
    now=$(expr $now + $interval) 
done

wait

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
    sleep 5
done
