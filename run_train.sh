#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25206
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

python train.py
