#!/bin/bash
#SBATCH --job-name=nyu_cancer_linear_ft_3chans_p12_ce_2xp_w_lr1e-5_100%_vit_seed_2004
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=a100_short
#SBATCH --output=../log/nyu_cancer_linear_ft_3chans_p12_ce_2xp_w_lr1e-5_100%_vit_seed_2004.out
#SBATCH --error=../log/nyu_cancer_linear_ft_3chans_p12_ce_2xp_w_lr1e-5_100%_vit_seed_2004.err

# activate conda env
source activate head_ct

# module load cuda/12.1
# module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12401 ../main_downstream.py --local_rank 0 \
    --model_name "vit" --batch_size 64  --num_workers 4 --max_epochs 10 --base_lr 1e-5 \
    --cfg ../configs/downstream/vit_HeadCT_nyu.yaml \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --preds_save_name "nyu_cancer_linear_ft_3chans_p12_2xp_w_ce_lr1e-5_100%_vit_seed_2004" --wandb_project "headct_ft" \
    --classifier "linear" --label_name "cancer" --dataset "nyu" --seed 2004   \
    --filename "nyu_cancer"