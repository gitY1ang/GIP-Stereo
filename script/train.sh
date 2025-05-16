#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -p gpu-a40
#SBATCH --gres=gpu:2

source /users/gzu_yang/miniconda3/bin/activate gip

CUDA_VISIBLE_DEVICES=0,1 python train_stereo.py --batch_size 8 --n_downsample 2 --num_steps 200000


