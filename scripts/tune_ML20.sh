#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --mem=84000M
#SBATCH --gres=gpu:2
#SBATCH --time=15-00:00:00

hostname
nvidia-smi


echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"

py=anaconda3/envs/jointeval/bin/python3.10
prg=DPFR-recsys-evaluation/RecBole/run_hyper_update.py

cd DPFR-recsys-evaluation/

model=$1

tunedata() {
    echo $1 $2
    if [ "$2" = "ItemKNN" ]; then 
        echo "model is ItemKNN"
        param=DPFR-recsys-evaluation/RecBole/hyperchoice/$2-$1.hyper
    else
        param=DPFR-recsys-evaluation/RecBole/hyperchoice/$2.hyper
    fi
    $py $prg \
                --dataset=$1 \
                --model=$2 \
                --params_file=$param \
                --gpu_id=${CUDA_VISIBLE_DEVICES:0:1}
                }

tunedata "new_ML-20M" $model

