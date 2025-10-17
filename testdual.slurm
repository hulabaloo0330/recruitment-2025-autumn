#!/bin/bash
#SBATCH -o %j.out			    #指定作业的标准输出
#SBATCH -e %j.err			    #指定作业的错误输出
#SBATCH -J gmres-2gpu		    #指定作业名称
#SBATCH -t 02:00:00			    #指定作业最大运行2小时
#SBATCH -N 1				    #指定节点数为 1
#SBATCH --gpus=2                #为作业申请 2 个 GPU

# 编译
# 假设 g++ 和 cuda 库在默认路径下
make clean && make

# 检查编译是否成功
if [ ! -f ./gmres ]; then
    echo "编译失败, 可执行文件 ./gmres 未找到。"
    exit 1
fi

# 矩阵文件所在的基础目录
MATRIX_DIR="/data/Recruitment/GMRES/sparseMatrixSet"
> gmres_time.txt # 清空日志

for i in $(seq 1 80)
do
    MTX_FILE="${MATRIX_DIR}/mtx${i}.bin"
    
    if [ -f "$MTX_FILE" ]; then
        echo "================================================="
        echo "Running GMRES for mtx${i}.bin"
        echo "================================================="
        # 直接运行可执行文件
        ./gmres "$MTX_FILE"
    else
        echo "Warning: Matrix file $MTX_FILE not found, skipping."
    fi
done

echo "所有测试完成。"