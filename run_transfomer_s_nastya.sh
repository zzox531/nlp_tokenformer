#!/bin/bash
# Transformer S
# training for HOURS hours. start with:  bash run_trf_s.sh
set -euo pipefail
mkdir -p logs checkpoints

# ── settings ──────────────────────────────────────────────────────────────
QOS="ik448298_a100"
ENV="tokenformer_entropy"
HOURS=7.0
SBATCH_TIME="07:30:00"          
# ───────────────────────────────────────────────────────────────────────────


NAME="trf_s"
MB=32          # micro-batch
GA=8           # grad accumulation
BIG_STEPS=100000   

JID=$(sbatch --parsable \
  --job-name="$NAME" \
  --partition=a100 --qos="$QOS" --gres=gpu:a100:1 --mem=48G \
  --time="$SBATCH_TIME" \
  --output="logs/%x_%j.out" --error="logs/%x_%j.err" \
  --wrap "#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda activate $ENV
export WANDB_MODE=offline && export HF_HOME=/tmp/\$USER/hf && mkdir -p \$HF_HOME
python train.py \
  --config config/${NAME}.json \
  --datasets Skylion007/openwebtext monology/pile-uncopyrighted --mix_probs 0.5 0.5 \
  --pile_data_dir data/pile \
  --seq_len 1024 \
  --batch_size $MB --grad_accum_steps $GA \
  --max_steps $BIG_STEPS --max_hours $HOURS \
  --lr 6e-4 --warmup_ratio 0.02 \
  --eval_interval 2000 --eval_tasks pile hellaswag \
  --eval_max_samples 500 --eval_pile_max_tokens 200000 \
  --save_dir checkpoints/${NAME} --wandb_project tf-compare")

echo "Submitted: Transformer S  ->  job $JID"
echo "Queue:  squeue --me     Log:  tail -f logs/${NAME}_${JID}.out"
echo "Checkpoint will be:  checkpoints/${NAME}/model_base.pt"