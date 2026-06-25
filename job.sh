#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanx:1
#SBATCH --time=00:15:00
#SBATCH --job-name=dermlip
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tokenformer_entropy

python dataloader.py --data_dir data/owt --split train
python dataloader.py --data_dir data/owt --split val

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
