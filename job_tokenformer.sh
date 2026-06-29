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

python dataloader.py --split val --data_dir data/pile_uncopyrighted \
    --dataset monology/pile-uncopyrighted --max_docs 2000

python train.py \
    --config config/mixtral_tokenformer.json \
    --datasets Skylion007/openwebtext monology/pile-uncopyrighted \
    --mix_probs 0.5 0.5 \
    --pile_data_dir data/pile_uncopyrighted \
    --seq_len 1024 \
    --batch_size 8 \
    --grad_accum_steps 16 \
    --lr 3e-4 \
    --max_steps 60000 \
    --eval_interval 250 \
    --save_dir checkpoints \
    --wandb_project mixtral-tokenformer
