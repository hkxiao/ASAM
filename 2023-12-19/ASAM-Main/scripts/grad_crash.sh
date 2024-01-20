#!/bin/bash
start_program() {
    export CUDA_VISIBLE_DEVICES=$1,$2
    python $3 \
    --save_root=output/sa_000000-Grad \
    --data_root=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --inversion_dir=output/Inversion/SSD-7.5-50-INV-5/embeddings \
    --control_mask_dir=sam-1b/sa_000000 \
    --SD_path=ckpt/SSD-1B \
    --SD_type=SSD \
    --ddim_steps=50 \
    --sam_batch=160 \
    --eps=0.2 \
    --steps=2 \
    --alpha=0.1 \
    --mu=0.5 \
    --beta=1.0 \
    --norm=2 \
    --gamma=32.0 \
    --kappa=32.0 \
    --start=$4 \
    --end=$5
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(0 2 4 6)
CUDA_VISIBLE_DEVICES_LIST2=(1 3 5 7)
start=(9191 9400 9700 10000)
end=(9400 9700 10000 11187)

PID_LIST=()
STATUS=()
Keyborad=false

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]} grad_null_text_inversion_edit.py ${start[i]} ${end[i]} &
    PID_LIST+=($!)
    STATUS+=(-1)
done

# check crash
while [ "$Keyboard" != "true" ];
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
        
            start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]} utils/grad_crash_aid.py ${start[i]} ${end[i]}
            STATUS[i]=$?
            echo "进程 $process_id 的退出状态为 ${STATUS[i]}"

            if [ ${STATUS[i]} -ne 0 ]; then
                start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} ${CUDA_VISIBLE_DEVICES_LIST2[i]} grad_null_text_inversion_edit.py ${start[i]} ${end[i]} &
                PID=$!
                PID_LIST[i]=($PID)
                echo "进程 ${PID_LIST[i]} 重新执行"
                finish=false
            fi
        fi
    done
    if [ $finish == true ]; then
        break
    fi
    sleep 30
done