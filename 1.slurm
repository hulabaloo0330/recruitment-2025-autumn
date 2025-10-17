#!/bin/bash
#SBATCH -o modules_%j.out
#SBATCH -e modules_%j.err
#SBATCH -J ModuleCheck
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --exclude=hepnode0
#SBATCH --gres=gpu:1
echo "--- 以下是运算节点 $(hostname) 上可用的模块 ---"

# 在运算节点上执行 module avail 命令
module avail