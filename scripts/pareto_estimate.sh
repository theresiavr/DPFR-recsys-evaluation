#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH -w hendrixgpu07fl
#SBATCH --ntasks=1 --mem=23000M
#SBATCH --time=1-00:00:00

hostname

echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"

py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation/pareto/estimate_pareto.py

cd DPFR-recsys-evaluation/

dataset=$1

echo $dataset

for i in {1..10}
do
   echo "Doing with $i desired points"
   $py $prg --dataset=$dataset --numpoint=$i
done

