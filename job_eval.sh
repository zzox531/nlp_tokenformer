#!/bin/bash
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=mixtral-base
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

export HF_TOKEN=<YOUR_HF_TOKEN>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tokenformer_entropy

python evaluate.py --checkpoint checkpoints/model_base.pt --tasks piqa
