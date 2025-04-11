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

from src.utils.misc import load_model, load_optimizer, init_distributed_mode, cleanup, \
    MultiCropWrapper, set_requires_grad_false, cosine_scheduler
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_lr_scheduler
from src.utils.wd_sched import get_wd_scheduler

from src.data.datasets import get_pretrain_dataloaders
from src.data.transforms import DataAugmentationDINO3D

from src.models.vit import ViT
from src.models.dino_head import DINOHead

from src.losses.losses import DINOLoss

from monai.config import print_config

from engine_pretrain_dino import *

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
    
    if config.MODEL.NAME == "dino":
        final_size = config.MODEL.ROI
        global_crops_size = config.DINO.GLOBAL_CROP_SIZE 
        local_crops_size = config.DINO.LOCAL_CROP_SIZE 
        local_crops_number = config.DINO.LOCAL_CROP_NUM
        # Define transforms for image
        imtrans = DataAugmentationDINO3D(final_size, global_crops_size, local_crops_size, local_crops_number)
        imvals = DataAugmentationDINO3D(final_size, global_crops_size, local_crops_size, local_crops_number)
        imtests = DataAugmentationDINO3D(final_size, global_crops_size, local_crops_size, local_crops_number)
    else:
        raise ValueError(f"Model {config.MODEL.NAME} not supported")
    
    augs = [imtrans, imvals, imtests]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_pretrain_dataloaders(config, augs)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and momentum model
    if config.MODEL.NAME == "dino":
        # encoder
        backbone = ViT(
            img_size=config.VIT.INPUT_SIZE,
            patch_size=config.VIT.PATCH_SIZE,
            hidden_size=config.VIT.HIDDEN_SIZE,
            mlp_dim=config.VIT.MLP_DIM,
            num_layers=config.VIT.NUM_LAYERS,
            num_heads=config.VIT.NUM_HEADS,
            in_chans=config.VIT.IN_CHANS,
            dropout_rate=config.VIT.DROPOUT_RATE,
            spatial_dims=config.VIT.SPATIAL_DIMS,
            num_register_tokens=config.VIT.NUM_REGISTER_TOKENS,
            patch_embed=config.VIT.PATCH_EMBED,
            pos_embed=config.VIT.POS_EMBED,
            classification=config.VIT.CLASSIFICATION,
            num_classes=config.DATA.NUM_CLASSES,
            post_activation="Tanh",
            qkv_bias=config.VIT.USE_BIAS,
        )
        # momentum encoder
        momentum_backbone = ViT(
            img_size=config.VIT.INPUT_SIZE,
            patch_size=config.VIT.PATCH_SIZE,
            hidden_size=config.VIT.HIDDEN_SIZE,
            mlp_dim=config.VIT.MLP_DIM,
            num_layers=config.VIT.NUM_LAYERS,
            num_heads=config.VIT.NUM_HEADS,
            in_chans=config.VIT.IN_CHANS,
            dropout_rate=config.VIT.DROPOUT_RATE,
            spatial_dims=config.VIT.SPATIAL_DIMS,
            num_register_tokens=config.VIT.NUM_REGISTER_TOKENS,
            patch_embed=config.VIT.PATCH_EMBED,
            pos_embed=config.VIT.POS_EMBED,
            classification=config.VIT.CLASSIFICATION,
            num_classes=config.DATA.NUM_CLASSES,
            post_activation="Tanh",
            qkv_bias=config.VIT.USE_BIAS,
        )
        # predictor
        predictor_dino = DINOHead(
            in_dim=config.VIT.HIDDEN_SIZE,
            out_dim=config.DINO.HEAD_N_PROTOTYPES,
            hidden_dim=config.DINO.HEAD_HIDDEN_DIM,
            bottleneck_dim=config.DINO.BOTTLENECK_DIM,
            nlayers=config.DINO.HEAD_N_LAYERS,
            use_bn=config.DINO.USE_BN,
            norm_last_layer=config.DINO.NORM_LAST_LAYER,
        )
        # momentum predictor
        momentum_predictor_dino = DINOHead(
            in_dim=config.VIT.HIDDEN_SIZE,
            out_dim=config.DINO.HEAD_N_PROTOTYPES,
            hidden_dim=config.DINO.HEAD_HIDDEN_DIM,
            bottleneck_dim=config.DINO.BOTTLENECK_DIM,
            nlayers=config.DINO.HEAD_N_LAYERS,
            use_bn=config.DINO.USE_BN,
            norm_last_layer=config.DINO.NORM_LAST_LAYER,
        )
    else:
        raise ValueError(f"Backbone {config.MODEL.NAME} not supported")
    
    # Multi-Crops Wrapper
    model = MultiCropWrapper(
        backbone=backbone, 
        head=predictor_dino,
    ).to(device)
    momentum_model = MultiCropWrapper(
        backbone=momentum_backbone, 
        head=momentum_predictor_dino,
    ).to(device)
    
    # Load model with wrong size weights unloaded
    all_state_dicts = load_model(config, model, momentum_model, logger)
    
    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    momentum_model = nn.SyncBatchNorm.convert_sync_batchnorm(momentum_model)
    
    # Use DistributedDataParallel for distributed training
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[device], 
        broadcast_buffers=False, 
        find_unused_parameters=True
    )
    momentum_model = nn.parallel.DistributedDataParallel(
        momentum_model, 
        device_ids=[device], 
        broadcast_buffers=False, 
        find_unused_parameters=True
    )
    
    # Set gradient computation on momentum model to false
    set_requires_grad_false(momentum_model)
    
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
    lr, min_lr = config.TRAIN.BASE_LR, config.TRAIN.MIN_LR
    optimizer = get_optimizer(config, lr, [model])
    lr_scheduler = get_lr_scheduler(config, optimizer, num_warmup_steps, total_steps, min_lr)
    wd_scheduler = get_wd_scheduler(config, len(train_loader))
    momentum_scheduler = cosine_scheduler(
        config.DINO.MOMENTUM_TEACHER, 
        config.DINO.MOMENTUM_TEACHER_END, 
        config.TRAIN.MAX_EPOCHS, 
        len(train_loader)
    )
    
    # Load optimizer & scheduler
    if config.MODEL.PRETRAINED:
        optimizer, lr_scheduler, start_epoch = load_optimizer(optimizer, lr_scheduler, all_state_dicts, logger)
    else:
        start_epoch = 0
    
    dino_loss = DINOLoss(
        out_dim=config.DINO.HEAD_N_PROTOTYPES,
        ncrops=config.DINO.LOCAL_CROP_NUM + 2,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp=config.DINO.WARMUP_TEACHER_TEMP, 
        teacher_temp=config.DINO.TEACHER_TEMP, 
        warmup_teacher_temp_epochs=config.DINO.WARMUP_TEACHER_EPOCHS, 
        nepochs=config.TRAIN.MAX_EPOCHS,
    ).cuda()
    
    # Training and testing
    if config.MODEL.NAME == "dino":
        train_loss = trainer(
            config=config,
            model=model,
            momentum_model=momentum_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            dino_criterion=dino_loss,
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            val_every=val_every,
            logger=logger,
            device=device,
            wandb_run=wandb_run,
        )
        logger.info(f"train completed, best train dino loss: {train_loss:.4f}")
        
        test_loss = tester(
            config=config,
            model=model,
            test_loader=test_loader,
            dino_criterion=dino_loss,
            momentum_model=momentum_model,
            logger=logger,
            device=device,
            wandb_run=wandb_run,
        )
        logger.info(f"test completed, best test dino loss: {test_loss:.4f}")
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
    args, config = parse_option()
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
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
            },
        )

    main(config, wandb_run)
