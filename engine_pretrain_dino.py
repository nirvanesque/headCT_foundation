import torch
import math
import time
import sys
import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from src.utils.misc import all_reduce_mean, save_checkpoint, MetricLogger, \
    cancel_gradients_last_layer, clip_gradients, _update_momentum_encoder
    

def train_one_epoch(
    config: Any,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    wd_scheduler: Any,
    momentum_scheduler: Any,
    epoch: int,
    max_epoch: int,
    dino_criterion: torch.nn.Module,
    momentum_model: Optional[torch.nn.Module] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        config (Any): Configuration object.
        model (torch.nn.Module): The main model.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (Any): Learning rate scheduler.
        wd_scheduler (Any): Weight decay scheduler.
        momentum_scheduler (Any): Momentum scheduler.
        epoch (int): Current epoch number.
        max_epoch (int): Maximum number of epochs.
        dino_criterion (torch.nn.Module): DINO loss criterion.
        momentum_model (Optional[torch.nn.Module]): Momentum model.
        logger (Optional[Any]): Logger.
        device (Optional[torch.device]): Device to use.
        use_amp (bool): Whether to use automatic mixed precision.
        scaler (Optional[torch.cuda.amp.GradScaler]): Gradient scaler for AMP.
        wandb_run (Optional[Any]): Weights and Biases run object.

    Returns:
        Dict[str, float]: Dictionary of average metrics.
    """
    model.train()
    momentum_model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    for idx, batch_data in enumerate(loader):
        # Weight decay scheduling
        it = len(loader) * epoch + idx  # Global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = wd_scheduler[it]
        
        # Zero gradients
        optimizer.zero_grad()
        loss = 0  # Initialize accumulated loss
        
        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in batch_data]
        
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            # Compute features for teacher and student models
            teacher_output = momentum_model(images[:2])
            student_output = model(images)
            
            # DINO output
            dino_teacher_output = teacher_output['dino_output']
            dino_student_output = student_output['dino_output']
            
            # Calculate DINO loss
            loss = dino_criterion(dino_student_output, dino_teacher_output, epoch)
            
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        if config.TRAIN.GRAD_CLIP:
            clip_gradients(model, config.TRAIN.GRAD_CLIP)
        
        # Cancel last layer gradient 
        cancel_gradients_last_layer(epoch, model, config.DINO.FREEZE_LAST_LAYER)
        
        # Update student model
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        
        # Update momentum (teacher) encoder
        with torch.no_grad():
            m = momentum_scheduler[idx]
            _update_momentum_encoder(model.module, momentum_model.module, m)
            
        torch.cuda.synchronize()
        
        # Reduce and log metrics
        loss_value = all_reduce_mean(loss)
        
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        wd = optimizer.param_groups[0]["weight_decay"]
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        
        logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        
        if wandb_run is not None and dist.get_rank() == 0:
            wandb_run.log({'Training Loss': float(loss_value), 'Training lr': lr, 'Training wd': wd})

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_one_epoch(
    config: Any,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    max_epoch: int,
    dino_criterion: torch.nn.Module,
    momentum_model: Optional[torch.nn.Module] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Validate the model for one epoch.

    Args:
        config (Any): Configuration object.
        model (torch.nn.Module): The main model.
        loader (torch.utils.data.DataLoader): DataLoader for validation data.
        epoch (int): Current epoch number.
        max_epoch (int): Maximum number of epochs.
        dino_criterion (torch.nn.Module): DINO loss criterion.
        momentum_model (Optional[torch.nn.Module]): Momentum model.
        logger (Optional[Any]): Logger.
        device (Optional[torch.device]): Device to use.
        use_amp (bool): Whether to use automatic mixed precision.
        scaler (Optional[torch.cuda.amp.GradScaler]): Gradient scaler for AMP.
        wandb_run (Optional[Any]): Weights and Biases run object.

    Returns:
        Dict[str, float]: Dictionary of average metrics.
    """
    # Loss weights
    
    model.eval()
    momentum_model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            loss = 0  # Initialize accumulated loss
            
            # Move images to GPU
            images = [im.cuda(non_blocking=True) for im in batch_data]
            
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                # Compute features for teacher and student models
                teacher_output = momentum_model(images[:2])
                student_output = model(images)
                    
                # DINO output
                dino_teacher_output = teacher_output['dino_output']
                dino_student_output = student_output['dino_output']

                # Calculate DINO loss
                loss = dino_criterion(dino_student_output, dino_teacher_output, epoch)
            
            torch.cuda.synchronize()
            
            # Reduce and log metrics
            loss_value = all_reduce_mean(loss)
            
            if not math.isfinite(loss_value):
                logger.info("Loss is {}, ignored".format(loss_value))
            
            metric_logger.update(loss=loss_value)
            
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
            
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def trainer(
    config: Any,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    wd_scheduler: Any,
    momentum_scheduler: Any,
    dino_criterion: torch.nn.Module,
    start_epoch: int = 0,
    max_epochs: int = 100,
    val_every: int = 10,
    momentum_model: Optional[torch.nn.Module] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    """
    Train the model.

    Args:
        config (Any): Configuration object.
        model (torch.nn.Module): The main model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (Any): Learning rate scheduler.
        wd_scheduler (Any): Weight decay scheduler.
        momentum_scheduler (Any): Momentum scheduler.
        dino_criterion (torch.nn.Module): DINO loss criterion.
        start_epoch (int): Starting epoch number.
        max_epochs (int): Maximum number of epochs.
        val_every (int): Validate every 'val_every' epochs.
        momentum_model (Optional[torch.nn.Module]): Momentum model.
        logger (Optional[Any]): Logger.
        device (Optional[torch.device]): Device to use.
        wandb_run (Optional[Any]): Weights and Biases run object.

    Returns:
        float: Best validation loss.
    """
    use_amp = config.AMP_ENABLE
    val_loss_min = float("inf")
    val_losses = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        
        # Train for one epoch
        train_stats = train_one_epoch(
            config,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            wd_scheduler,
            momentum_scheduler,
            epoch,
            max_epochs,
            dino_criterion,
            momentum_model=momentum_model,
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
        
        # Save latest checkpoint
        if dist.get_rank() == 0:
            save_checkpoint(
                model,
                momentum_model,
                epoch,
                optimizer,
                scheduler=lr_scheduler,
                filename='last_' + config.MODEL.SAVE_NAME,
                best_loss=val_loss_min,
                dir_add=config.MODEL.DIR,
                logger=logger,
            )
        
        # Validate every 'val_every' epochs
        if (epoch + 1) % val_every == 0 and epoch != 0:
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                val_loader,
                epoch,
                max_epochs,
                dino_criterion,
                momentum_model=momentum_model,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                wandb_run=wandb_run,
            )
            
            logger.info(
                f"Final validation {epoch+1}/{max_epochs} \
                    loss: {val_stats['loss']}, time {time.time() - epoch_time}s"
            )
            
            if wandb_run is not None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss': float(val_stats['loss'])})
            
            val_losses.append(val_stats['loss'])
            
            # Save best checkpoint
            if val_stats['loss'] < val_loss_min:
                logger.info(f"new best ({val_loss_min} --> {val_stats['loss']}). ")
                val_loss_min = val_stats['loss']
                if dist.get_rank() == 0:
                    save_checkpoint(
                        model,
                        momentum_model,
                        epoch,
                        optimizer,
                        scheduler=lr_scheduler,
                        filename='best_' + config.MODEL.SAVE_NAME,
                        best_loss=val_loss_min,
                        dir_add=config.MODEL.DIR,
                        logger=logger,
                    )
                
    logger.info(f"Training Finished !, Best Loss: {val_loss_min}")
    return val_loss_min

def tester(
    config: Any,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    dino_criterion: torch.nn.Module,
    momentum_model: Optional[torch.nn.Module] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    """
    Test the model.

    Args:
        config (Any): Configuration object.
        model (torch.nn.Module): The main model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        dino_criterion (torch.nn.Module): DINO loss criterion.
        momentum_model (Optional[torch.nn.Module]): Momentum model.
        logger (Optional[Any]): Logger.
        device (Optional[torch.device]): Device to use.
        wandb_run (Optional[Any]): Weights and Biases run object.

    Returns:
        float: Test loss.
    """
    epoch_time = time.time()
    use_amp = config.AMP_ENABLE
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epoch, max_epoch = 0, 1

    # Validate the model
    test_stats = val_one_epoch(
        config,
        model,
        test_loader,
        epoch,
        max_epoch,
        dino_criterion,
        momentum_model=momentum_model,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        wandb_run=wandb_run,
    )
    
    logger.info(
        f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s"
    )

    if wandb_run is not None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']
