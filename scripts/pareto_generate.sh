#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH -w hendrixgpu07fl
#SBATCH --ntasks=1 --mem=23000M
#SBATCH --time=1-00:00:00

hostname


py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation/pareto/generate_pareto.py

cd DPFR-recsys-evaluation/

dataset=$1

echo $dataset

$py $prg --dataset=$dataset

