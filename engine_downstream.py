import math
import time
import sys
import copy
import pickle
import wandb
import logging

from typing import Tuple, List, Optional, Any

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
import torch.distributed as dist

from src.utils.misc import all_reduce_mean, plot_regression, \
    plot_pr_curve, save_checkpoint, MetricLogger

def train_one_epoch(
    config: object,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    optimizers: List[Optimizer],
    schedulers: List[_LRScheduler],
    criterion: torch.nn.Module,
    epoch: int,
    max_epoch: int,
    train_metric_collection: MetricCollection,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> dict:
    """
    Train the model for one epoch.
    
    Args:
        config (object): Configuration object containing training parameters.
        model (torch.nn.Module): The model to be trained.
        classifier (torch.nn.Module): The classifier to be trained.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizers (list): List of optimizers for training.
        schedulers (list): List of learning rate schedulers.
        criterion (torch.nn.Module): Loss function.
        epoch (int): Current epoch number.
        max_epoch (int): Maximum number of epochs.
        train_metric_collection (MetricCollection): Collection of metrics to compute during training.
        logger (logging.Logger, optional): Logger for logging information. Defaults to None.
        device (torch.device, optional): Device to run the training on. Defaults to None.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision training. Defaults to None.
        wandb_run (wandb.Run, optional): Weights and Biases run object for logging. Defaults to None.
    
    Returns:
        dict: Dictionary containing average metrics for the epoch.
    """
    model_name = config.MODEL.NAME
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    all_logits, all_targets = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    for idx, batch_data in enumerate(loader):
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        data, target, _ = batch_data
        data, target = data.to(device), target.to(device)
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            if model_name == 'vit':
                out, _ = model(data)
            else:
                raise NotImplementedError(f"Unknown model: {model_name}")
            
            # Use [CLS] embedding for linear classifier
            if config.TRAIN.CLASSIFIER == 'linear' and model_name == 'vit':
                out = out[:, :1, :].squeeze()
            
            # Linear or attentive probing
            logits = classifier(out)
            
            # Compute loss
            loss = criterion(logits, target)
        
        # Collect metrics
        if config.DATA.NUM_CLASSES != 1:
            all_logits = torch.cat((all_logits, F.softmax(logits, dim=1).clone().detach()), dim=0)
            all_targets = torch.cat((all_targets, target.clone().detach().long()), dim=0)
        else:
            raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
            
        # Backpropagation
        scaler.scale(loss).backward()
        
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        
        # Gradient clipping
        if config.TRAIN.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.TRAIN.GRAD_CLIP)
            if not config.TRAIN.LOCK:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRAD_CLIP)
        
        # Optimizers update
        for optimizer in optimizers:
            scaler.step(optimizer)
            
        scaler.update()
        
        # Schedulers update
        for scheduler in schedulers:
            scheduler.step()
            
        torch.cuda.synchronize()
        loss_value = all_reduce_mean(loss)
        
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        lr = optimizers[0].param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        if wandb_run != None and dist.get_rank() == 0:
            wandb_run.log({'Training Loss': float(loss_value), 'Training lr': lr})

    train_metric_collection(all_logits, all_targets.long())
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    config: object,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    epoch: int,
    max_epoch: int,
    val_metric_collection: MetricCollection,
    criterion: torch.nn.Module, 
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    save_preds: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    """
    Validate the model for one epoch.
    
    Args:
        config (object): Configuration object containing validation parameters.
        model (torch.nn.Module): The model to be validated.
        classifier (torch.nn.Module): The classifier to be validated.
        loader (torch.utils.data.DataLoader): DataLoader for validation data.
        epoch (int): Current epoch number.
        max_epoch (int): Maximum number of epochs.
        val_metric_collection (MetricCollection): Collection of metrics to compute during validation.
        criterion (torch.nn.Module): Loss function.
        logger (logging.Logger, optional): Logger for logging information. Defaults to None.
        device (torch.device, optional): Device to run the validation on. Defaults to None.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        save_preds (bool, optional): Whether to save predictions. Defaults to False.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision validation. Defaults to None.
    
    Returns:
        dict: Dictionary containing average metrics for the epoch.
    """
    all_preds, all_targets = [], []
    fnames = []
    
    model_name = config.MODEL.NAME
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    all_logits = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target, fname = batch_data
            data, target = data.to(device), target.to(device)
            
            # Mixed precision evaluation
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                if save_preds:
                    fnames += fname
                    
                if model_name == 'vit':
                    out, _ = model(data)
                else:
                    raise NotImplementedError(f"Unknown model: {model_name}")
                
                # Use [CLS] embedding for linear classifier
                if config.TRAIN.CLASSIFIER == 'linear' and model_name == 'vit':
                    out = out[:, :1, :].squeeze()
                
                # Linear or attentive probing
                logits = classifier(out)
                
                # Compute loss
                loss = criterion(logits, target)
                all_preds.append(logits.detach().cpu())
                all_targets.append(target.detach().cpu())
            
            torch.cuda.synchronize()
            # Collect metrics
            if config.DATA.NUM_CLASSES != 1:
                all_logits = torch.cat((all_logits, F.softmax(logits, dim=1).clone().detach()), dim=0)
            else:
                raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
            
            loss_value = all_reduce_mean(loss)
            
            metric_logger.update(loss=loss_value)
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    preds_save_name = config.PREDS_SAVE_NAME
    
    # Save predictions to pickle
    if save_preds:
        save_dict = {'fnames': fnames, 'preds': F.softmax(all_preds.float(), dim=1)[:, 1].numpy(), 'targets': all_targets.numpy()}
        with open(f'../preds_pkl/{preds_save_name}_preds.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
    
    # Plot results
    if config.DATA.NUM_CLASSES == 2:
        all_preds = F.softmax(all_preds.float(), dim=1)[:, 1].numpy()
        plot_pr_curve(all_targets.numpy(), all_preds, preds_save_name)
    else:
        raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
    
    val_metric_collection(all_logits, all_targets.to(device).long())
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    config: object,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: List[Optimizer],
    schedulers: List[_LRScheduler],
    criterion: torch.nn.Module,
    start_epoch: int = 0,
    max_epochs: int = 100,
    val_every: int = 10,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> Tuple[float, torch.nn.Module, torch.nn.Module]:
    r"""
    Trains and validates a model and classifier.
    Args:
        config (object): Configuration object containing training parameters.
        model (torch.nn.Module): The model to be trained.
        classifier (torch.nn.Module): The classifier to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizers (list): List of optimizers for training.
        schedulers (list): List of learning rate schedulers.
        criterion (torch.nn.Module): Loss function.
        start_epoch (int, optional): Starting epoch for training. Defaults to 0.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 100.
        val_every (int, optional): Frequency of validation (in epochs). Defaults to 10.
        logger (logging.Logger, optional): Logger for logging information. Defaults to None.
        device (torch.device, optional): Device to run the training on. Defaults to None.
        wandb_run (wandb.Run, optional): Weights and Biases run object for logging. Defaults to None.
    Returns:
        Tuple: Best validation AUROC, best model, and best classifier.
    """
    best_classifier = copy.deepcopy(classifier)
    if not config.TRAIN.LOCK:
        best_model = copy.deepcopy(model)
    else:
        best_model = model
        
    use_amp = config.AMP_ENABLE
    
    val_auroc_max = -1
    val_losses = []
    
    if config.DATA.NUM_CLASSES != 1:
        train_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.DATA.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.DATA.NUM_CLASSES, average=None),
        ]).to(device)
        
        val_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.DATA.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.DATA.NUM_CLASSES, average=None),
        ]).to(device)
    else:
        raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        
        train_stats = train_one_epoch(
            config,
            model,
            classifier,
            train_loader,
            optimizers,
            schedulers,
            criterion,
            epoch,
            max_epochs,
            train_metric_collection,
            logger=logger,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            wandb_run=wandb_run,
        )
        logger.info(f"Final training  {epoch+1}/{max_epochs}, loss: {train_stats['loss']}, time {time.time() - epoch_time}s")
        
        metric_out = train_metric_collection.compute()
        
        if config.DATA.NUM_CLASSES != 1:
            acc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy()
            auroc_out = metric_out["MulticlassAUROC"].detach().cpu().numpy()
            logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
        else:
            raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
            
        train_metric_collection.reset()

        if (epoch + 1) % val_every == 0 and (val_every == 1 or epoch != 0):
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                classifier,
                val_loader,
                epoch,
                max_epochs,
                val_metric_collection,
                criterion,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            logger.info(f"Final validation {epoch+1}/{max_epochs} loss: {val_stats['loss']}, time {time.time() - epoch_time}s")
            metric_out = val_metric_collection.compute()
            
            if config.DATA.NUM_CLASSES != 1:
                acc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy()
                auroc_out = metric_out["MulticlassAUROC"].detach().cpu().numpy()
                logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
            else:
                raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
                
            val_metric_collection.reset()
            
            if wandb_run is not None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss': float(val_stats['loss'])})
            
            val_losses.append(val_stats['loss'])
            val_auroc = sum(auroc_out) / len(auroc_out)
            
            if val_auroc > val_auroc_max:
                logger.info(f"new best AUROC ({val_auroc_max} --> {val_auroc}). ")
                val_auroc_max = val_auroc
                if dist.get_rank() == 0:
                    save_checkpoint(
                        model,
                        None,
                        epoch,
                        optimizers[0],
                        schedulers[0],
                        best_loss=val_auroc,
                        dir_add=config.MODEL.DIR,
                        filename=config.MODEL.SAVE_NAME,
                        logger=logger,
                    )
                    classifier_save_name = config.MODEL.SAVE_NAME.split('.')[0] + '_classifier' + '.pt'
                    save_checkpoint(
                        classifier,
                        None, 
                        epoch,
                        optimizers[0],
                        schedulers[0],
                        best_loss=val_auroc,
                        dir_add=config.MODEL.DIR,
                        filename=classifier_save_name,
                        logger=logger,
                    )
                best_classifier = copy.deepcopy(classifier)
                if not config.TRAIN.LOCK:
                    best_model = copy.deepcopy(model)
                
    logger.info(f"Training Finished !, Best AUROC: {val_auroc_max}")
    
    return val_auroc_max, best_model, best_classifier


def tester(
    config: object,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    r"""
    Test the given model using the provided configuration and data loader.
    Args:
        config (object): Configuration object containing various settings.
        model (torch.nn.Module): The model to be tested.
        classifier (torch.nn.Module): The classifier to be used.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        logger (logging.Logger, optional): Logger for logging information. Defaults to None.
        device (torch.device, optional): Device to run the model on. Defaults to None.
        wandb_run (wandb.Run, optional): Weights and Biases run object for logging. Defaults to None.
    Returns:
        float: The final test loss.
    """
    epoch_time = time.time()
    
    use_amp = config.AMP_ENABLE
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if config.DATA.NUM_CLASSES != 1:
        test_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.DATA.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.DATA.NUM_CLASSES, average=None),
        ]).to(device)
    else:
        raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")

    epoch, max_epoch = 0, 1

    test_stats = val_one_epoch(
        config,
        model,
        classifier,
        test_loader,
        epoch,
        max_epoch,
        test_metric_collection,
        criterion,
        logger=logger,
        device=device,
        use_amp=use_amp,
        save_preds=True,
        scaler=scaler,
    )
    
    logger.info(f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s")
    
    metric_out = test_metric_collection.compute()
    
    if config.DATA.NUM_CLASSES != 1:
        acc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy()
        auroc_out = metric_out["MulticlassAUROC"].detach().cpu().numpy()
        logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
    else:
        raise NotImplementedError(f"Unknown number of classes: {config.DATA.NUM_CLASSES}")
        
    test_metric_collection.reset()

    if wandb_run is not None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']
