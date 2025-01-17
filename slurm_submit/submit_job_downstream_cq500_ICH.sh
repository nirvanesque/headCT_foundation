#!/bin/bash
#SBATCH --job-name=vit_ICH_linear_prob_3chans_p12_ce_2xp_w_lr1e-5_100%_scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=a100_short
#SBATCH --output=../log/vit_cq500_ICH_linear_prob_3chans_p12_ce_2xp_w_lr1e-5_100%_scratch.out
#SBATCH --error=../log/vit_cq500_ICH_linear_prob_3chans_p12_ce_2xp_w_lr1e-5_100%_scratch.err

# activate conda env
source activate head_ct

# module load cuda/12.1
# module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ../main_downstream.py --local_rank 0 \
    --model_name "vit" --batch_size 64 --num_workers 4 --max_epochs 10 --base_lr 1e-5 \
    --cfg ../configs/downstream/vit_HeadCT_cq500.yaml \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --preds_save_name "cq500_ICH_linear_prob_3chans_p12_2xp_w_ce_lr1e-5_100%_scratch" --use_amp --wandb_project "headct_ft" \
    --classifier "linear" --label_name "ICH" --dataset "cq500" --seed 2004   \
    --filename "cq500_ICH"