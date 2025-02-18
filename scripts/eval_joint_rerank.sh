#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH -w hendrixgpu03fl
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

hostname

echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"

py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation/eval/eval_joint_rerank.py

cd DPFR-recsys-evaluation/

dataset=$1
echo "all reranked models and all datasets"

$py $prg 

