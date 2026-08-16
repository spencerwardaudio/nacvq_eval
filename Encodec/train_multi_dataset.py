import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path
import numpy as np

import hydra
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import wandb

import multi_dataset as data
from multi_dataset import collate_fn
from losses import disc_loss, total_loss
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed)
from balancer import Balancer
from cal_metrics import calculate_si_snr

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def calculate_confidence_interval(values, confidence=0.95):
    """Calculate mean and 95% confidence interval for a list of values."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    n = len(values)
    
    # Calculate 95% confidence interval using t-distribution
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_val * (std / np.sqrt(n))
    
    return mean, mean - margin_error, mean + margin_error


def train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader, config, scheduler, disc_scheduler, scaler=None, scaler_disc=None, balancer=None, wandb_logger=None):
    """Train one step function with bandwidth logging."""
    model.train()
    disc_model.train()
    data_length = len(trainloader)
    
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0

    for idx, input_wav in enumerate(trainloader):
        input_wav = input_wav.contiguous()
        if torch.cuda.is_available():
            input_wav = input_wav.cuda()
        optimizer.zero_grad()
        
        with autocast(enabled=config.common.amp):
            output, loss_w, _ = model(input_wav)
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            )
        
        if config.common.amp: 
            loss_g = (3*losses_g['l_g'] + 3*losses_g['l_feat']
                      + losses_g['l_t']/10 + losses_g['l_f']
                      + loss_w)
            scaler.scale(loss_g).backward()  
            scaler.step(optimizer)  
            scaler.update()   
            scheduler.step()  
        else:
            if balancer is not None:
                balancer.backward(losses_g, output, retain_graph=True)
                loss_g = sum([l * balancer.weights.get(k, 0) for k, l in losses_g.items()])
            else:
                loss_g = (3*losses_g['l_g'] + 3*losses_g['l_feat']
                          + losses_g['l_t']/10 + losses_g['l_f'])
                loss_g.backward(retain_graph=True)
            loss_w.backward()
            optimizer.step()

        # Accumulate losses  
        accumulated_loss_g += loss_g.item()
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += loss_w.item()

        # Update discriminator with probability
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config.model.train_discriminator 
                               and epoch >= config.lr_scheduler.warmup_epoch])
        if torch.cuda.is_available():
            train_discriminator = train_discriminator.cuda()

        if train_discriminator:
            with autocast(enabled=config.common.amp):
                logits_real, _ = disc_model(input_wav)
                logits_fake, _ = disc_model(output.detach())
                loss_disc = disc_loss(logits_real, logits_fake)
            if config.common.amp: 
                scaler_disc.scale(loss_disc).backward()
                scaler_disc.step(optimizer_disc)  
                scaler_disc.update()  
            else:
                loss_disc.backward() 
                optimizer_disc.step()

            accumulated_loss_disc += loss_disc.item()
        
        scheduler.step()
        disc_scheduler.step()

    # Print epoch summary
    avg_loss_g = accumulated_loss_g / data_length
    avg_loss_w = accumulated_loss_w / data_length
    avg_loss_disc = accumulated_loss_disc / data_length if accumulated_loss_disc > 0 else 0.0
    
    log_msg = f"| TRAIN | epoch: {epoch} | loss_g: {avg_loss_g:.4f} | loss_w: {avg_loss_w:.4f} | lr_G: {optimizer.param_groups[0]['lr']:.6e} | lr_D: {optimizer_disc.param_groups[0]['lr']:.6e}"
    
    if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
        log_msg += f" | loss_disc: {avg_loss_disc:.4f}"
    
    logger.info(log_msg)
    
    # Weights & Biases logging
    if wandb_logger:
        log_dict = {
            'epoch': epoch,
            'train/loss_g': avg_loss_g,
            'train/loss_w': avg_loss_w,
            'train/lr_g': optimizer.param_groups[0]['lr'],
            'train/lr_d': optimizer_disc.param_groups[0]['lr'],
        }
        for k, l in accumulated_losses_g.items():
            log_dict[f'train/{k}'] = l / data_length
        if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
            log_dict['train/loss_disc'] = avg_loss_disc
        wandb_logger.log(log_dict)


@torch.no_grad()
def validate(epoch, model, disc_model, valloader, config, wandb_logger=None):
    """Validation function with bandwidth-specific metrics."""
    model.eval()
    disc_model.eval()
    
    total_loss_g = 0.0
    total_loss_disc = 0.0
    total_si_snr = 0.0
    num_samples = 0
    
    # Track metrics per bandwidth - store individual values for confidence intervals
    bandwidth_metrics = defaultdict(lambda: {
        'si_snr': [], 'count': 0
    })
    
    for idx, input_wav in enumerate(valloader):
        if torch.cuda.is_available():
            input_wav = input_wav.cuda()
        
        # Test all bandwidths
        for bandwidth in config.model.target_bandwidths:
            model.bandwidth = bandwidth
            output = model(input_wav)
            
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            loss_disc = disc_loss(logits_real, logits_fake)
            losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
            
            total_loss_g += sum([l.item() for l in losses_g.values()])
            total_loss_disc += loss_disc.item()
            
            # Calculate comprehensive metrics for each sample in batch
            batch_size = input_wav.shape[0]
            for i in range(batch_size):
                ref_sample = input_wav[i].squeeze().cpu()
                rec_sample = output[i].squeeze().cpu()
                
                # Calculate only SI-SNR metric
                try:
                    si_snr = calculate_si_snr(ref_sample.numpy(), rec_sample.numpy())
                except Exception as e:
                    si_snr = None
                
                # Update totals and collect individual values
                if si_snr is not None:
                    total_si_snr += si_snr
                    bandwidth_metrics[bandwidth]['si_snr'].append(si_snr)
                
                bandwidth_metrics[bandwidth]['count'] += 1
                num_samples += 1
    
    avg_loss_g = total_loss_g / (len(valloader) * len(config.model.target_bandwidths))
    avg_loss_disc = total_loss_disc / (len(valloader) * len(config.model.target_bandwidths))
    avg_si_snr = total_si_snr / num_samples if num_samples > 0 else 0.0
    
    log_msg = f"| VAL  | epoch: {epoch} | loss_g: {avg_loss_g:.4f} | loss_disc: {avg_loss_disc:.4f} | SI-SNR: {avg_si_snr:.2f} dB"
    logger.info(log_msg)
    
    # Log bandwidth-specific metrics with confidence intervals
    for bandwidth, metrics in bandwidth_metrics.items():
        count = metrics['count']
        if count > 0:
            # Calculate confidence intervals
            si_snr_mean, si_snr_ci_low, si_snr_ci_high = calculate_confidence_interval(metrics['si_snr'])
            
            logger.info(f"  Bandwidth {bandwidth} kbps (n={count}):")
            si_snr_margin = (si_snr_ci_high - si_snr_ci_low) / 2
            logger.info(f"    SI-SNR: {si_snr_mean:.2f}±{si_snr_margin:.2f} dB")
    
    # Weights & Biases logging
    if wandb_logger:
        val_log_dict = {
            'epoch': epoch,
            'val/loss_g': avg_loss_g,
            'val/loss_disc': avg_loss_disc,
            'val/si_snr': avg_si_snr,
        }
        
        # Log bandwidth-specific metrics with confidence intervals
        for bandwidth, metrics in bandwidth_metrics.items():
            count = metrics['count']
            if count > 0:
                si_snr_mean, si_snr_ci_low, si_snr_ci_high = calculate_confidence_interval(metrics['si_snr'])
                
                val_log_dict[f'val/si_snr_bw_{bandwidth}'] = si_snr_mean
                val_log_dict[f'val/si_snr_bw_{bandwidth}_ci_low'] = si_snr_ci_low
                val_log_dict[f'val/si_snr_bw_{bandwidth}_ci_high'] = si_snr_ci_high
        
        wandb_logger.log(val_log_dict)

    return avg_loss_g


def train(config):
    """Main training function."""
    # Remove existing logging handlers
    logger.handlers.clear()

    # Set up logging
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_multi_dataset_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # Print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Initialize Weights & Biases
    wandb_logger = None
    if config.get('wandb', {}).get('enabled', True):
        try:
            wandb.init(
                project=config.get('wandb', {}).get('project', 'multi-dataset-encodec'),
                name=config.get('wandb', {}).get('name', f'multi_dataset_bs{config.datasets.batch_size}_epochs{config.common.max_epoch}'),
                config=dict(config),
                dir=config.checkpoint.save_folder,
            )
            wandb_logger = wandb
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            logger.info("✓ Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            wandb_logger = None

    # Set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # Set up datasets
    trainset = data.MultiDataset(config=config, mode='train')
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory
    )
    
    
    # Set up models
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, 
        model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(str(config.model.segment)) if config.model.segment is not None else None, 
        name=config.model.name,
        ratios=config.model.ratios,
        use_q2d2=config.model.get('use_q2d2', False),
        q2d2_grid=config.model.get('q2d2_grid', 'rhombic'),
        q2d2_levels=config.model.get('q2d2_levels', 11),
    )
    
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    # Log model information
    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    logger.info(f"Encodec Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"Model train mode: {model.training} | Quantizer train mode: {model.quantizer.training}")
    logger.info(f"Target bandwidths: {config.model.target_bandwidths}")

    # Resume training if specified
    resume_epoch = 0
    if config.checkpoint.resume:
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        if not os.path.exists(config.checkpoint.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {config.checkpoint.checkpoint_path}")
        if not os.path.exists(config.checkpoint.disc_checkpoint_path):
            raise FileNotFoundError(f"Discriminator checkpoint not found: {config.checkpoint.disc_checkpoint_path}")

        logger.info(f"Loading model checkpoint from: {config.checkpoint.checkpoint_path}")
        logger.info(f"Loading discriminator checkpoint from: {config.checkpoint.disc_checkpoint_path}")

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        
        model.load_state_dict(model_checkpoint['model_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        logger.info(f"✓ Successfully loaded checkpoints and resuming from epoch {resume_epoch}")

    if torch.cuda.is_available():
        model.cuda()
        disc_model.cuda()

    # Note: valloader is created inside the training loop on first validation
    logger.info(f"Training: {len(trainloader)} batches with mixing strategy (batch_size={config.datasets.batch_size})")
    logger.info(f"Validation: {config.datasets.n_val_segments} mixed segments per epoch with mixing strategy (batch_size={config.datasets.batch_size})")

    # Set up optimizers and schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params': disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    
    scheduler = WarmupCosineLrScheduler(
        optimizer, 
        max_iter=config.common.max_epoch * len(trainloader), 
        eta_ratio=0.1, 
        warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), 
        warmup_ratio=1e-4
    )
    disc_scheduler = WarmupCosineLrScheduler(
        optimizer_disc, 
        max_iter=config.common.max_epoch * len(trainloader), 
        eta_ratio=0.1, 
        warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), 
        warmup_ratio=1e-4
    )

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    # Load optimizer and scheduler states if resuming
    if config.checkpoint.resume:
        if 'optimizer_state_dict' in model_checkpoint.keys():
            optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
            logger.info(f"✓ Loaded generator optimizer state from epoch {resume_epoch}")
        
        if 'optimizer_state_dict' in disc_model_checkpoint.keys():
            optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
            logger.info(f"✓ Loaded discriminator optimizer state from epoch {resume_epoch}")
        
        if 'scheduler_state_dict' in model_checkpoint.keys():
            scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Loaded generator scheduler state from epoch {resume_epoch}")
            
        if 'scheduler_state_dict' in disc_model_checkpoint.keys():
            disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Loaded discriminator scheduler state from epoch {resume_epoch}")

    start_epoch = max(1, resume_epoch + 1)
    best_val_loss = float('inf')

    # Instantiate loss balancer
    balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    
    # Training loop
    for epoch in range(start_epoch, config.common.max_epoch + 1):
        train_one_step(
            epoch, optimizer, optimizer_disc, 
            model, disc_model, trainloader, config,
            scheduler, disc_scheduler, scaler, scaler_disc, balancer, wandb_logger
        )
        
        # Validation
        if epoch % config.common.val_interval == 0 and epoch > 0:
            # Create validation dataset with fixed segments for consistent evaluation across epochs
            valset = data.MultiDataset(config=config, mode='val')
            valloader = torch.utils.data.DataLoader(
                valset,
                batch_size=config.datasets.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config.datasets.num_workers,
                pin_memory=config.datasets.pin_memory
            )
            val_loss = validate(epoch, model, disc_model, valloader, config, wandb_logger)
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f'{config.checkpoint.save_location}best_model.pt'
                best_disc_path  = f'{config.checkpoint.save_location}best_disc.pt'
                save_master_checkpoint(epoch, model, optimizer, scheduler, best_model_path)
                save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, best_disc_path)
                logger.info(f'[BEST] val loss {val_loss:.4f} at epoch {epoch} — saved best_model.pt, best_disc.pt')


        # Save checkpoint
        if epoch % config.common.save_interval == 0:
            if not os.path.exists(config.checkpoint.save_folder):
                os.makedirs(config.checkpoint.save_folder)
            model_path = f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt'
            disc_path = f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt'
            
            save_master_checkpoint(epoch, model, optimizer, scheduler, model_path)  
            save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, disc_path)
            
            # Log model artifacts to wandb
            if wandb_logger:
                artifact = wandb.Artifact(f'model_epoch_{epoch}', type='model')
                artifact.add_file(model_path)
                artifact.add_file(disc_path)
                wandb_logger.log_artifact(artifact)
    
    # Finish wandb run
    if wandb_logger:
        wandb.finish()


@hydra.main(config_path='config', config_name='config_multi_dataset')
def main(config):
    # Disable cudnn
    torch.backends.cudnn.enabled = False
    
    # Memory optimization
    torch.cuda.empty_cache()
    # NOTE: expandable_segments not supported in PyTorch < 2.2, removed for compatibility
    
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    
    train(config)


if __name__ == '__main__':
    main()
