#!/bin/bash

# 设置GPU列表

CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
# CUDA_VISIBLE_DEVICES_LIST=(1 3 4 5 6 7)

#CUDA_VISIBLE_DEVICES_LIST=(0)

# start=(1 500 1000 1500 2000 2500 3000 3500)
# end=(500 1000 1500 2000 2500 3000 3500 4000)

# start=(1)
# end=(500)


now=11187
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

subset="sa_000001"

start_program() {
    export CUDA_VISIBLE_DEVICES=$1
    
    echo $5
    
    python $2 \
        --save_root=output/${subset}-Grad \
        --data_root=sam-1b/${subset} \
        --control_mask_dir=sam-1b/${subset} \
        --caption_path=sam-1b/${subset}-blip2-caption.json \
        --inversion_dir=output/${subset}-Direction-Inversion/embeddings \
        --inversion_type direct_inversion \
        --controlnet_path=ckpt/control_v11p_sd15_mask_sa000001.pth \
        --sd_path=/data/vjuicefs_ai_camera_jgroup_research/public_data/11170238/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9 \
        --prompt_bs=4 \
        --eps=0.2 \
        --eps_boxes=20.0 \
        --eps_points=20.0 \
        --ddim_steps=50 \
        --steps=10 \
        --alpha=0.01 \
        --alpha_boxes=4 \
        --alpha_points=4 \
        --boxes_noise_scale=0.1 \
        --points_noise_scale=0.1 \
        --mu=0.5 \
        --beta=0.5 \
        --norm=2 \
        --gamma=100 \
        --kappa=100 \
        --attack_object image points boxes \
        --prompt_type  points boxes \
        --embedding_sup True \
        --embedding_mse_weight 0.5 \
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
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} grad_null_text_inversion_edit_prompt.py ${start[i]} ${end[i]} &

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
            start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} grad_null_text_inversion_edit_prompt.py ${start[i]} ${end[i]} "--check_crash"

            STATUS[i]=$?
            echo "process $process_id exiting status is ${STATUS[i]}"

            if [ ${STATUS[i]} -ne 0 ]; then
                start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} grad_null_text_inversion_edit_prompt.py ${start[i]} ${end[i]}&
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