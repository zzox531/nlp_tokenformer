#!/bin/bash
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=mixtral-tokenformer
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tokenformer_entropy

python train.py \
    --config config/mixtral_tokenformer.json \
    --data_dir data/owt \
    --seq_len 512 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --lr 3e-4 \
    --max_steps 10000 \
    --save_dir checkpoints \
    --wandb_project mixtral-tokenformer
