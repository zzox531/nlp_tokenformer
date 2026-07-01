#!/bin/bash
# tokenformer S , tokenformer S + MoE.
# every train job runs for HOURS hours (default 6). Launch:  bash run_two_nastya.sh
# QOS limit = 2 jobs, one GPU -> they will run one after another automatically.
set -euo pipefail
mkdir -p logs checkpoints

# ── settings ──────────────────────────────────────────────────────────────
QOS="ik448298_a100"
ENV="tokenformer_entropy"
HOURS=7.0                       
SBATCH_TIME="07:30:00"          
# ───────────────────────────────────────────────────────────────────────────

declare -A MB GA
MB[tkf_s]=32;    GA[tkf_s]=8
MB[tkfmoe_s]=24; GA[tkfmoe_s]=10

BIG_STEPS=100000

submit() {
  local name="$1"
  sbatch --parsable \
    --job-name="$name" \
    --partition=a100 --qos="$QOS" --gres=gpu:a100:1 --mem=48G \
    --time="$SBATCH_TIME" \
    --output="logs/%x_%j.out" --error="logs/%x_%j.err" \
    --wrap "#!/bin/bash
            . ~/miniconda3/etc/profile.d/conda.sh && conda activate $ENV && \
            export WANDB_MODE=offline && export HF_HOME=/tmp/\$USER/hf && mkdir -p \$HF_HOME && \
            python train.py \
              --config config/${name}.json \
              --datasets Skylion007/openwebtext monology/pile-uncopyrighted --mix_probs 0.5 0.5 \
              --pile_data_dir data/pile \
              --seq_len 1024 \
              --batch_size ${MB[$name]} --grad_accum_steps ${GA[$name]} \
              --max_steps $BIG_STEPS --max_hours $HOURS \
              --lr 6e-4 --warmup_ratio 0.02 \
              --eval_interval 2000 --eval_tasks pile hellaswag \
              --eval_max_samples 500 --eval_pile_max_tokens 200000 \
              --save_dir checkpoints/${name} --wandb_project tf-compare"
}

#J1=$(submit tkf_s)
#echo "Submitted: tokenformer S        -> job $J1"
#J2=$(submit tkfmoe_s)
#echo "Submitted: tokenformer S + MoE  -> job $J2 (will wait for GPU to be available)"
#echo "Done. Queue:  squeue --me     Log:  tail -f logs/tkf_s_${J1}.out"


J2=$(submit tkfmoe_s)
echo "Submitted: tokenformer S + MoE  -> job $J2(will wait for GPU to be available)"
echo "Done. Queue:  squeue --me     Log:  tail -f logs/tkfmoe_s_${J2}.out"