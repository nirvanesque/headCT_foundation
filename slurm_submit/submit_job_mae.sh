#!/bin/bash
#SBATCH --job-name=mae_final_lr1.5e-4_mask0.75_sincos_pflash_ep400_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=radiology
#SBATCH --output=../log/mae_final_lr1.5e-4_mask0.75_sincos_pflash_ep400_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42.out
#SBATCH --error=../log/mae_final_lr1.5e-4_mask0.75_sincos_pflash_ep400_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42.err

# activate conda env
source activate head_ct

# module load cuda/12.1
# module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ../main_pretrain_mae.py --local_rank 0 \
    --model_name "mae" --batch_size 256 --num_workers 4 --max_epochs 400 --base_lr 1.5e-4 \
    --cfg ../configs/mae/mae_HeadCT.yaml \
    --use_amp --optimizer "AdamW" --scheduler "cosine" --weight_decay 5e-3 \
    --grad_clip 3.0 --use_wandb
