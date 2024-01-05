#!/bin/bash
start_program() {
    export CUDA_VISIBLE_DEVICES=$1 
    python $2 \
    --save_root=output/Inversion \
    --data_root=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --guidance_scale=7.5 \
    --steps=5 \
    --ddim_steps=50 \
    --start=$3 \
    --end=$4  
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
start=(4000 4500 5000 5500 6000 6500 7000 7500)
end=(5400 5000 5500 6000 6500 7000 7500 8000)

PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} null_text_inversion.py ${start[i]} ${end[i]} &
    PID_LIST+=($!)
    STATUS+=(-1)
done

# check crash
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