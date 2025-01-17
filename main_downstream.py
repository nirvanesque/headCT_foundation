import os
import json
import random
import warnings
import argparse
import wandb
import pickle
import numpy as np
from logger import create_logger
from config import get_config

from src.utils.misc import load_model, init_distributed_mode, cleanup, set_requires_grad_false
from src.utils.pos_embed import interpolate_pos_embed
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_lr_scheduler

from src.data.transforms import vit_transforms
from src.data.datasets import get_finetune_dataloaders, get_fewshots_dataloaders

from src.models.layers import RMSNorm
from src.models.vit import ViT
from src.models.classifier import LinearClassifier, AttentionClassifier

from monai.config import print_config

from engine_downstream import *

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
    
    # 
    parser.add_argument("--preds_save_name", type=str, help='save name tag for predictions')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist-backend', default='nccl', help='used to set up distributed backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--seed", type=int, help='seed')
    parser.add_argument("--use_amp", action='store_true')

    # wandb configs
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--filename", type=str, default="monai-test")
    parser.add_argument("--wandb_project", type=str, default="monai-test")

    # model parameters
    parser.add_argument("--model_name", type=str, help='model name')
    parser.add_argument("--model_load_path", type=str, help='path to trained model')
    parser.add_argument("--classifier", type=str, help='classifier name (linear or attentive)')
    parser.add_argument("--label_name", type=str, help='label name for downstream tasks')
    parser.add_argument("--optimizer", type=str, help='training optimizer')
    parser.add_argument("--scheduler", type=str, help='learning rate scheduler')
    parser.add_argument("--base_lr", type=float, help='base learning rate')
    parser.add_argument("--min_lr", type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", type=float, help='max epoch')
    parser.add_argument("--grad_clip", type=float, help='gradient clipping')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--num_workers", type=int, help='number of workers for dataloader')
    parser.add_argument("--max_epochs", type=int, help='max epoch')
    parser.add_argument("--lock", action='store_true')

    # dataset parameters
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--train_csv_path', type=str, help='path to train csv file')
    parser.add_argument('--val_csv_path', type=str, help='path to val csv file')
    parser.add_argument('--test_csv_path', type=str, help='path to test csv file')
    parser.add_argument("--few_shots", type=int, help='number of few shots')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config, wandb_run):
    # Define parameters
    max_epochs = config.TRAIN.MAX_EPOCHS
    val_every = config.TRAIN.VAL_EVERY
    
    # Create data transformations
    imtrans = vit_transforms(config, mode='train')
    imvals = vit_transforms(config, mode='val')
    imtests = vit_transforms(config, mode='test')
    augs = [imtrans, imvals, imtests]
    
    # Create dataloaders
    if config.DATA.FEW_SHOTS == -1:
        train_loader, val_loader, test_loader, class_weights = get_finetune_dataloaders(config, augs)
    else:
        train_loader, val_loader, test_loader, class_weights = get_fewshots_dataloaders(config, augs)
    
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    if config.MODEL.NAME == "vit":
        if config.MAE.NORM_LAYER == 'layernorm':
            norm_layer = nn.LayerNorm
        elif config.MAE.NORM_LAYER == 'rmsnorm':
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Normalization layer {config.MAE.NORM_LAYER} not supported")
        
        # Define backone model
        model = ViT(
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
            lora=config.TRAIN.LORA,
            norm_layer=norm_layer,
        ).to(device)
        
        # Define Classifier
        if config.TRAIN.CLASSIFIER == 'linear':
            classifier = LinearClassifier(
                dim=config.VIT.HIDDEN_SIZE, 
                num_classes=config.DATA.NUM_CLASSES
            ).to(device)
        elif config.TRAIN.CLASSIFIER == 'attentive':
            classifier = AttentionClassifier(
                dim=config.VIT.HIDDEN_SIZE, 
                num_classes=config.DATA.NUM_CLASSES, 
                num_heads=12, 
                num_queries=1
            ).to(device)
        else:
            raise ValueError(f"Classifier {config.TRAIN.CLASSIFIER} not supported")
    else:
        raise ValueError(f"Backbone {config.MODEL.NAME} not supported")
    
    # Load model with wrong size weights unloaded
    load_model(config, model, None, logger)

    # Compile model and classifier
    model, classifier = torch.compile(model), torch.compile(classifier)

    # Lock backbone if required
    if config.TRAIN.LOCK:
        set_requires_grad_false(model)

    # Lock LoRA parameters if required
    if config.TRAIN.LORA:
        set_requires_grad_false(model, lora=config.TRAIN.LORA)
        
    # Count and print the number of trainable parameters
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()  # Number of elements in the tensor
            total_params += num_params
    logger.info(f"Total trainable parameters: {total_params}")
    
    torch.backends.cudnn.benchmark = True
    
    # Calculate effective total training steps
    world_size = dist.get_world_size()
    effective_batch_size = config.DATA.BATCH_SIZE * world_size
    total_steps = len(train_loader) * config.TRAIN.MAX_EPOCHS
    # Calculate warmup steps
    num_warmup_steps = int(config.TRAIN.PER_WARMUP * total_steps)
    
    # Learning rate scaling
    config.defrost()
    config.TRAIN.MIN_LR = config.TRAIN.BASE_LR * 1e-3
    config.freeze()

    logger.info(f"Effective Learning Rate: {config.TRAIN.BASE_LR}, Effective Batch Size: {effective_batch_size}, Max Epochs: {config.TRAIN.MAX_EPOCHS}")
    logger.info(f"Number of Warmup Steps: {num_warmup_steps}, Total Steps: {total_steps}")

    # Initialize optimizers
    lr = config.TRAIN.BASE_LR
    optimizer_model = get_optimizer(config, lr, [model])
    optimizer_classifier = get_optimizer(config, lr * 1e2, [classifier])
    
    optimizers = [optimizer_classifier] if config.TRAIN.LOCK else [optimizer_model, optimizer_classifier]
    
    # Initialize schedulers
    min_lr_model = config.TRAIN.MIN_LR
    scheduler_model = get_lr_scheduler(config, optimizer_model, num_warmup_steps, total_steps, min_lr_model)
    
    min_lr_classifier = config.TRAIN.MIN_LR * 1e2
    scheduler_classifier = get_lr_scheduler(config, optimizer_classifier, num_warmup_steps, total_steps, min_lr_classifier)
    
    schedulers = [scheduler_classifier] if config.TRAIN.LOCK else [scheduler_model, scheduler_classifier]
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define start epoch
    start_epoch = 0

    # Train the model
    train_loss, best_model, best_classifier = trainer(
        config=config,
        model=model,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        criterion=criterion,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        val_every=val_every,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )

    logger.info(f"train completed, best train loss: {train_loss:.4f} ")

    # Test the model
    test_loss = tester(
        config=config,
        model=best_model,
        classifier=best_classifier,
        test_loader=test_loader,
        criterion=criterion,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )
    
    logger.info(f"test completed, best test loss: {test_loss:.4f} ")
    cleanup()

def init_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == "__main__":
    # Deprecate warning
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
    
    # Parse config options
    args, config = parse_option()
    
    # Initialize distributed training
    init_distributed_mode(args)
    
    # Define and initialize seed
    seed = config.SEED + dist.get_rank()
    init_seed(seed)
    
    # Create logger
    logger = create_logger(output_dir=config.LOG.OUTPUT_DIR, dist_rank=dist.get_rank(), name=config.LOG.FILENAME)

    # Output config settings
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, f"{config.LOG.FILENAME}.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # Print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    # Initialize wandb
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

    # Run main training
    main(config, wandb_run)
