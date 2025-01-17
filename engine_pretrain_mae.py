import torch
import math
import time
import sys
import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from src.utils.misc import all_reduce_mean, save_checkpoint, MetricLogger, \
    clip_gradients


def train_one_epoch(
    config: Any,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    max_epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        config: Configuration object.
        model: The model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        max_epoch: Maximum number of epochs.
        logger: Logger for logging information.
        device: Device to run the training on.
        use_amp: Whether to use automatic mixed precision.
        scaler: GradScaler for mixed precision training.
        wandb_run: Weights and Biases run object for logging.

    Returns:
        Dictionary with average loss and learning rate.
    """
    model_name = config.MODEL.NAME
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    for idx, batch_data in enumerate(loader):
        optimizer.zero_grad()
        
        if model_name == 'mae':
            data = batch_data.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                loss, _, _ = model(data)
        else:
            raise NotImplementedError(f"Unknown model: {model_name}")
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        if config.TRAIN.GRAD_CLIP:
            clip_gradients(model, config.TRAIN.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        torch.cuda.synchronize()
        loss_value = all_reduce_mean(loss)
        
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        if wandb_run is not None and dist.get_rank() == 0:
            wandb_run.log({'Training Loss': float(loss_value), 'Training lr': lr})

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    config: Any,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    max_epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Validate the model for one epoch.

    Args:
        config: Configuration object.
        model: The model to validate.
        loader: DataLoader for validation data.
        epoch: Current epoch number.
        max_epoch: Maximum number of epochs.
        logger: Logger for logging information.
        device: Device to run the validation on.
        use_amp: Whether to use automatic mixed precision.
        scaler: GradScaler for mixed precision validation.

    Returns:
        Dictionary with average loss.
    """
    model_name = config.MODEL.NAME
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if model_name == 'mae':
                data = batch_data.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                    loss, _, _ = model(data)
            else:
                raise NotImplementedError(f"Unknown model: {model_name}")
            
            loss_value = all_reduce_mean(loss)
            
            if not math.isfinite(loss_value):
                logger.info(f"Loss is {loss_value}, ignored")
            
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    config: Any,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int = 0,
    max_epochs: int = 100,
    val_every: int = 10,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    """
    Train the model for a specified number of epochs.

    Args:
        config: Configuration object.
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        start_epoch: Starting epoch number.
        max_epochs: Maximum number of epochs.
        val_every: Validate every 'val_every' epochs.
        logger: Logger for logging information.
        device: Device to run the training on.
        wandb_run: Weights and Biases run object for logging.

    Returns:
        Best validation loss achieved during training.
    """
    use_amp = config.AMP_ENABLE
    val_loss_min = float("inf")
    val_losses = []
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        train_stats = train_one_epoch(
            config,
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            max_epochs,
            logger=logger,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            wandb_run=wandb_run,
        )
        logger.info(
            f"Final training  {epoch+1}/{max_epochs}, loss: {train_stats['loss']}, \
                time {time.time() - epoch_time}s"
        )
        
        if dist.get_rank() == 0:
            save_checkpoint(
                model,
                None,
                epoch,
                optimizer,
                scheduler,
                best_loss=val_loss_min,
                dir_add=config.MODEL.DIR,
                filename='latest_' + config.MODEL.SAVE_NAME,
                logger=logger,
            )
        
        if (epoch + 1) % val_every == 0 and epoch != 0:
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                val_loader,
                epoch,
                max_epochs,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            logger.info(
                f"Final validation {epoch+1}/{max_epochs} \
                    loss: {val_stats['loss']}, time {time.time() - epoch_time}s"
            )
            
            if wandb_run is not None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss': float(val_stats['loss'])})
            
            val_losses.append(val_stats['loss'])
            
            if val_stats['loss'] < val_loss_min:
                logger.info(f"new best ({val_loss_min} --> {val_stats['loss']}). ")
                val_loss_min = val_stats['loss']
                if dist.get_rank() == 0:
                    save_checkpoint(
                        model,
                        None,
                        epoch,
                        optimizer,
                        scheduler,
                        best_loss=val_loss_min,
                        dir_add=config.MODEL.DIR,
                        filename='best_' + config.MODEL.SAVE_NAME,
                        logger=logger,
                    )
                
    logger.info(f"Training Finished !, Best Loss: {val_loss_min}")
    
    return val_loss_min


def tester(
    config: Any,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    """
    Test the model on the test dataset.

    Args:
        config: Configuration object.
        model: The model to test.
        test_loader: DataLoader for test data.
        logger: Logger for logging information.
        device: Device to run the testing on.
        wandb_run: Weights and Biases run object for logging.

    Returns:
        Test loss.
    """
    epoch_time = time.time()
    use_amp = config.AMP_ENABLE
    scaler = torch.amp.GradScaler(enabled=use_amp)
    epoch, max_epoch = 0, 1

    test_stats = val_one_epoch(
        config,
        model,
        test_loader,
        epoch,
        max_epoch,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    
    logger.info(
        f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s"
    )

    if wandb_run is not None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']