#!/bin/bash

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
# CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3)
# CUDA_VISIBLE_DEVICES_LIST=(0 2 4 7)
# CUDA_VISIBLE_DEVICES_LIST=(0)

# start=(1 500 1000 1500 2000 2500 3000 3500)
# end=(500 1000 1500 2000 2500 3000 3500 4000)

# start=(1)
# end=(500)

now=22375
interval=1400
start=()
end=()

for ((i=0; i<${#CUDA_VISIBLE_DEVICES_LIST[@]};i++)); do
    start+=($now)
    end+=($((now + interval)))
    now=$((now + interval))
done

echo "start&end list:"
echo ${start[@]} 
echo ${end[@]}

# kill $$
subset="sa_000002"

#--controlnet_path=ckpt/controlnet_plus_plus \

start_program() {
    export CUDA_VISIBLE_DEVICES=$1

    echo $5
    echo "GPU" $1
    
    python $2 \
        --save_root=output/${subset}-Direction-Inversion \
        --data_root=sam-1b/${subset} \
        --control_mask_dir=sam-1b/${subset} \
        --caption_path=sam-1b/${subset}-blip2-caption.json \
        --controlnet_path=ckpt/control_v11p_sd15_mask_sa000001.pth \
        --sd_path=/data/vjuicefs_ai_camera_jgroup_research/public_data/11170238/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9 \
        --inversion_type direct_inversion \
        --guidence_scale=7.5 \
        --steps=10 \
        --ddim_steps=50 \
        --start=$3 \
        --end=$4 \
        $5
}


PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} inversion.py ${start[i]} ${end[i]} &

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
            echo "process ${process_id} is finished"
            continue
        elif ps -p $process_id > /dev/null; then
            echo "process ${process_id} is executing"
            finish=false
        else
            echo "process ${process_id} is not executing"
            
            # 获取进程的退出状态
            start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} inversion.py ${start[i]} ${end[i]} "--check_crash"

            STATUS[i]=$?
            echo "process $process_id exiting status is ${STATUS[i]}"

            if [ ${STATUS[i]} -ne 0 ]; then
                start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} inversion.py ${start[i]} ${end[i]}&
                PID_LIST[i]=$!
                echo "process ${PID_LIST[i]} executes again"
                finish=false
            fi
        fi
    done
    if [ $finish == true ]; then
        break
    fi
    sleep 5
done
