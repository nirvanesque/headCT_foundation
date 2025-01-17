import os
import copy
import json
import random
import warnings
import argparse
import wandb
import numpy as np
from logger import create_logger
from config import get_config

from src.utils.misc import load_model, load_optimizer, init_distributed_mode, cleanup
from src.utils.pos_embed import interpolate_pos_embed
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_lr_scheduler

from src.data.datasets import get_pretrain_dataloaders
from src.data.transforms import mae3d_transforms

from src.models.layers import RMSNorm
from src.models.mae import MaskedAutoencoderViT

from monai.config import print_config

from engine_pretrain_mae import *

import torch
import torch.nn as nn
import torch.distributed as dist

print_config()

def parse_option():
    parser = argparse.ArgumentParser('MONAI training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs='+',
    )

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist-backend', default='nccl', help='used to set up distributed backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--seed", type=int, help='seed')
    parser.add_argument("--use_amp", action='store_true')

    # wandb configs
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="monai-test")

    # model parameters
    parser.add_argument("--model_name", type=str, help='model name')
    parser.add_argument("--model_load_path", type=str, help='path to trained model')
    parser.add_argument("--optimizer", type=str, help='training optimizer')
    parser.add_argument("--scheduler", type=str, help='learning rate scheduler')
    parser.add_argument("--base_lr", type=float, help='base learning rate')
    parser.add_argument("--min_lr", type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", type=float, help='max epoch')
    parser.add_argument("--grad_clip", type=float, help='gradient clipping')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--num_workers", type=int, help='number of workers for dataloader')
    parser.add_argument("--max_epochs", type=int, help='max epoch')

    # dataset parameters
    parser.add_argument('--train_csv_path', type=str, help='path to train csv file')
    parser.add_argument('--val_csv_path', type=str, help='path to val csv file')
    parser.add_argument('--test_csv_path', type=str, help='path to test csv file')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, wandb_run):
    # Define parameters
    max_epochs = config.TRAIN.MAX_EPOCHS
    val_every = config.TRAIN.VAL_EVERY
    
    if config.MODEL.NAME == "mae":
        # Define transforms for image and segmentation
        imtrans = mae3d_transforms(config, mode='train', reshape=True)
        imvals = mae3d_transforms(config, mode='val', reshape=True)
        imtests = mae3d_transforms(config, mode='test', reshape=True)
    else:
        raise ValueError(f"Model {config.MODEL.NAME} not supported")
    
    augs = [imtrans, imvals, imtests]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_pretrain_dataloaders(config, augs)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model
    if config.MODEL.NAME == "mae":
        norm_layer = nn.LayerNorm if config.MAE.NORM_LAYER == 'layernorm' else RMSNorm
        model = MaskedAutoencoderViT(
            input_size=config.MAE.INPUT_SIZE,
            patch_size=config.MAE.PATCH_SIZE,
            mask_ratio=config.MAE.MASK_RATIO,
            in_chans=config.MAE.IN_CHANS,
            dropout_rate=config.MAE.DROPOUT_RATE,
            spatial_dims=config.MAE.SPATIAL_DIMS,
            patch_embed=config.MAE.PATCH_EMBED,
            pos_embed=config.MAE.POS_EMBED,
            encoder_depth=config.MAE.ENCODER_DEPTH,
            encoder_embed_dim=config.MAE.ENCODER_EMBED_DIM,
            encoder_mlp_dim=config.MAE.ENCODER_MLP_DIM,
            encoder_num_heads=config.MAE.ENCODER_NUM_HEADS,
            decoder_depth=config.MAE.DECODER_DEPTH,
            decoder_embed_dim=config.MAE.DECODER_EMBED_DIM,
            decoder_mlp_dim=config.MAE.DECODER_MLP_DIM,
            decoder_num_heads=config.MAE.DECODER_NUM_HEADS,
            norm_pix_loss=config.MAE.NORM_PIX_LOSS,
            use_bias=config.MAE.USE_BIAS,
            norm_layer=norm_layer,
        ).to(device)
    else:
        raise ValueError(f"Model {config.MODEL.NAME} not supported")

    # Load pretrained model if available
    if config.MODEL.PRETRAINED:
        loaded_state_dict = torch.load(config.MODEL.PRETRAINED, map_location=torch.device('cpu'))
        model_state_dict = loaded_state_dict['state_dict']
        new_model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        interpolate_pos_embed(model, new_model_state_dict)
        msg = model.load_state_dict(new_model_state_dict, strict=False)
        logger.info(f"Load Pretrained Model: {msg} for Architecture: {config.MODEL.NAME}")
        
    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Use DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
    
    torch.backends.cudnn.benchmark = True
    
    world_size = dist.get_world_size()
    effective_batch_size = config.DATA.BATCH_SIZE * world_size
    total_steps = len(train_loader) * config.TRAIN.MAX_EPOCHS
    num_warmup_steps = int(config.TRAIN.PER_WARMUP * total_steps)
    
    # Learning rate scaling
    config.defrost()
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * effective_batch_size / 256
    config.TRAIN.MIN_LR = config.TRAIN.BASE_LR * 1e-3
    config.freeze()
    
    logger.info(f"Effective Learning Rate: {config.TRAIN.BASE_LR}, Effective Batch Size: {effective_batch_size}, Max Epochs: {config.TRAIN.MAX_EPOCHS}")
    logger.info(f"Number of Warmup Steps: {num_warmup_steps}, Total Steps: {total_steps}")
    
    # Create optimizer & scheduler
    optimizer = get_optimizer(config, config.TRAIN.BASE_LR, [model])
    scheduler = get_lr_scheduler(config, optimizer, num_warmup_steps, total_steps, config.TRAIN.MIN_LR)
    
    # Load optimizer & scheduler if pretrained model is available
    if config.MODEL.PRETRAINED:
        optimizer, scheduler, start_epoch = load_optimizer(optimizer, scheduler, loaded_state_dict, logger)
    else:
        start_epoch = 0
    
    # Training and testing
    if config.MODEL.NAME == "mae":
        train_loss = trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            val_every=val_every,
            logger=logger,
            device=device,
            wandb_run=wandb_run,
        )
        logger.info(f"Train completed, best train reconstruction loss: {train_loss:.4f}")

        test_loss = tester(
            config=config,
            model=model,
            test_loader=test_loader,
            logger=logger,
            device=device,
            wandb_run=wandb_run,
        )
        logger.info(f"Test completed, best test reconstruction loss: {test_loss:.4f}")
    else:
        raise ValueError(f"Model {config.MODEL.NAME} not supported")

    cleanup()

def init_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
    args, config = parse_option()
    init_distributed_mode(args)
    seed = config.SEED + dist.get_rank()
    init_seed(seed)
    logger = create_logger(output_dir=config.LOG.OUTPUT_DIR, dist_rank=dist.get_rank(), name=config.LOG.FILENAME)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, f"{config.LOG.FILENAME}.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    wandb_run = None
    if config.WANDB.WANDB_ENABLE and dist.get_rank() == 0:
        wandb_run = wandb.init(
            name=config.LOG.FILENAME,
            project=config.WANDB.PROJECT,
            config={
                "learning_rate": config.TRAIN.BASE_LR,
                "batch_size": config.DATA.BATCH_SIZE,
                "epochs": config.TRAIN.MAX_EPOCHS,
                "backbone": config.MODEL.NAME,
            }
        )

    main(config, wandb_run)
