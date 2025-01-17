#!/bin/bash
#SBATCH --job-name=dino_final_lr5e-4_sincos_pflash_ep100_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42_wd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=50G
##SBATCH --mem=512G
#SBATCH --partition=radiology
#SBATCH --output=../log/dino_final_lr5e-4_sincos_pflash_ep100_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42_wd.out
#SBATCH --error=../log/dino_final_lr5e-4_sincos_pflash_ep100_adamw_clip3.0_reshape_layernorm_p12_channel3_gpu4_s42_wd.err

# activate conda env
source activate head_ct

# module load cuda/12.1
# module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 4 --master_port 12400 ../main_pretrain_dino.py --local_rank 0 \
    --model_name "dino" --batch_size 64 --num_workers 16 --max_epochs 200 --base_lr 5e-5 \
    --cfg ../configs/dino/dino_HeadCT.yaml \
    --use_amp --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.04 \
    --grad_clip 3.0 --use_wandb
