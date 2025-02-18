#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH -w hendrixgpu03fl
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

hostname


py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation//eval/eval_joint.py

cd DPFR-recsys-evaluation/

dataset=$1
echo "eval_joint.py"
echo "all non-reranked models and all datasets"

$py $prg 
