#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --mem=23000M
#SBATCH --time=15-00:00:00

hostname


py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation/eval/eval_MME_rerank.py

cd DPFR-recsys-evaluation/

dataset=$1
echo $dataset

$py $prg --dataset=$dataset
