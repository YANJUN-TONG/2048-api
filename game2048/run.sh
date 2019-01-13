#!/bin/bash
#SBATCH -J tr2048_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0 cuda/9.0 cudnn/7.0.5
source activate tf
python -u train.py 
