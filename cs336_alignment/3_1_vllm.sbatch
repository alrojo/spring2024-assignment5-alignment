#!/bin/bash
#SBATCH -J 3_1_vllm
#SBATCH -o 3_1_vllm_%j.out
#SBATCH -e 3_1_vllm_%j.err
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --time 10:00
#SBATCH -C GPU_SKU:H100_SXM5

module load cuda/12.6.1
module load cudnn/9.4.0
module load gcc/14.2.0
export CUDA_HOME=/share/software/user/open/cuda/12.6.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LLAMA_STACK_CONFIG_DIR==/scratch/users/arjo/.llama
export HF_HOME=/scratch/users/arjo/.huggingface
export HF_TOKEN=$HF_HOME/token
eval "$(conda shell.bash hook)"
conda activate cs336_alignment
python /scratch/users/arjo/spring2024-assignment5-alignment/cs336_alignment/3_1_vllm.py
