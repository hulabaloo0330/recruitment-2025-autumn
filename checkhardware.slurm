#!/bin/bash
#SBATCH -o cluster_check_sorted_%j.out
#SBATCH -e cluster_check_sorted_%j.err
#SBATCH -J ClusterCheckSorted
#SBATCH -t 00:05:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1

srun bash -c '
HOST=$(hostname)
CPU_MODEL=$(lscpu | grep "Model name:" | sed "s/Model name: *//")
echo "[$HOST] CPU Model: $CPU_MODEL"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | while IFS=, read -r INDEX NAME; do
        TRIMMED_NAME=$(echo "$NAME" | xargs)
        echo "[$HOST] GPU $INDEX: $TRIMMED_NAME"
    done
else
    echo "[$HOST] GPU Info: Not found"
fi

    if command -v nvidia-smi &> /dev/null
    then
        nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | while IFS=, read -r INDEX NAME; do
            TRIMMED_NAME=$(echo $NAME | xargs)
            echo "[$HOST] GPU $INDEX: $TRIMMED_NAME"
        done
    else
        echo "[$HOST] GPU Info: Not found"
    fi

    # --- 新增：CUDA 版本查询 ---
    # 我们直接检查你之前找到的 nvcc 路径
    NVCC_PATH="/usr/local/cuda-12/bin/nvcc"
    if [ -x "$NVCC_PATH" ]
    then
        # 如果 nvcc 存在且可执行，就运行它来获取版本信息
        # 我们用 grep "release" 来筛选出最关键的那一行
        CUDA_VERSION=$("$NVCC_PATH" --version | grep "release")
        echo "[$HOST] CUDA Version: $CUDA_VERSION"
    else
        echo "[$HOST] CUDA Version: nvcc not found at $NVCC_PATH"
    fi
' | sort