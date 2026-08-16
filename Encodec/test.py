#!/usr/bin/env python3
"""
Combined test script that:
1. Loads trained Encodec model and discriminator checkpoints
2. Processes all demo folders with ground truth audio files
3. Reconstructs audio at multiple bandwidths
4. Calculates metrics (SI-SNR)
5. Creates spectrograms
6. Logs everything to WandB in organized tables
"""

import os
import argparse
import torch
import torchaudio
import wandb
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from utils import convert_audio, save_audio
from cal_metrics import calculate_si_snr


def load_model_and_discriminator(model_checkpoint_path, disc_checkpoint_path, device='cuda'):
    """
    Load both the Encodec model and discriminator from checkpoints.
    
    Args:
        model_checkpoint_path (str): Path to the model checkpoint
        disc_checkpoint_path (str): Path to the discriminator checkpoint  
        device (str): Device to load models on
        
    Returns:
        tuple: (model, disc_model) loaded models
    """
    print(f"Loading model checkpoint from: {model_checkpoint_path}")
    print(f"Loading discriminator checkpoint from: {disc_checkpoint_path}")
    
    # Load model checkpoint
    model_checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    disc_model_checkpoint = torch.load(disc_checkpoint_path, map_location='cpu')
    
    # Create model with same config as training
    target_bandwidths = [1.5, 3., 6., 12., 24.]
    sample_rate = 24_000
    channels = 1
    
    model = EncodecModel._get_model(
        target_bandwidths=target_bandwidths,
        sample_rate=sample_rate,
        channels=channels,
        causal=True,
        model_norm='weight_norm',
        audio_normalize=True,
        segment=None,
        name='multi_dataset_encodec',
        ratios=[8, 5, 4, 2]
    )
    
    # Create discriminator with same config as training
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=channels,
        out_channels=channels,
        filters=32,
        hop_lengths=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        n_ffts=[2048, 1024, 512, 256, 128],
    )
    
    # Load state dicts
    model.load_state_dict(model_checkpoint['model_state_dict'])
    disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    disc_model.eval()
    
    # Move to device
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        disc_model = disc_model.cuda()
        print(f"Models loaded on CUDA")
    else:
        print(f"Models loaded on CPU")
    
    print(f"✓ Successfully loaded both checkpoints")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in disc_model.parameters()):,}")
    
    return model, disc_model


def reconstruct_audio(model, input_wav_path, bandwidth=6.0, device='cuda'):
    """
    Reconstruct audio from input wav file using the loaded model.
    
    Args:
        model: Loaded Encodec model
        input_wav_path (str): Path to input wav file
        bandwidth (float): Target bandwidth for reconstruction
        device (str): Device to run inference on
        
    Returns:
        torch.Tensor: Reconstructed audio tensor [C, T]
    """
    # Load input audio
    wav, sr = torchaudio.load(input_wav_path)
    
    # Convert audio to model format
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    # Add batch dimension: [C, T] -> [1, C, T]
    wav = wav.unsqueeze(0)
    
    # Set target bandwidth
    if bandwidth not in model.target_bandwidths:
        print(f"Warning: Bandwidth {bandwidth} not in {model.target_bandwidths}, using {model.target_bandwidths[0]}")
        bandwidth = model.target_bandwidths[0]
    
    model.set_target_bandwidth(bandwidth)
    
    # Move to device
    if torch.cuda.is_available() and device == 'cuda':
        wav = wav.cuda()
    
    # Reconstruct audio
    with torch.no_grad():
        reconstructed = model(wav)
    
    # Remove batch dimension for saving: [1, C, T] -> [C, T]
    reconstructed = reconstructed.squeeze(0)
    
    return reconstructed


def create_and_save_spectrogram(audio_file, output_dir, sr=24000):
    """Create a spectrogram image and save it to the output directory."""
    try:
        # Load audio
        audio, _ = librosa.load(audio_file, sr=sr)
        
        # Create spectrogram
        S = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(f'Spectrogram: {Path(audio_file).name}', fontsize=12)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        
        # Add colorbar
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Create output filename
        audio_file_path = Path(audio_file)
        spectrogram_name = audio_file_path.stem + '_spectrogram.png'
        spectrogram_path = output_dir / spectrogram_name
        
        # Save the spectrogram
        plt.savefig(spectrogram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return spectrogram_path
    except Exception as e:
        print(f"Error creating spectrogram for {audio_file}: {e}")
        return None


def process_demo_folders(model, disc_model, demo_dir, device='cuda', 
                         bandwidths=[1.5, 3., 6., 12., 24.], project_name="multi-dataset-encodec", 
                         entity=None):
    """
    Process all demo folders: reconstruct audio, calculate metrics, create spectrograms,
    and log to WandB.
    """
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        print(f"Demo directory not found: {demo_path}")
        return
    
    # Initialize WandB run
    run = wandb.init(
        project=project_name,
        entity=entity,
        name="test-audio-reconstruction",
        job_type="testing"
    )
    
    print(f"Processing demo folders from: {demo_path}")
    print(f"Saving reconstructed audio directly in demo folders")
    
    # Find all demo folders
    demo_folders = [f for f in demo_path.iterdir() if f.is_dir() and f.name != "README.md"]
    demo_folders.sort()
    
    print(f"Found {len(demo_folders)} demo folders:")
    for folder in demo_folders:
        print(f"  - {folder.name}")
    
    # Process each demo folder
    all_listening_tables = {}
    
    for folder in demo_folders:
        print(f"\n{'='*60}")
        print(f"Processing {folder.name}...")
        print(f"{'='*60}")
        
        # Find ground truth audio file
        gt_files = list(folder.glob("*_gt.wav"))
        if not gt_files:
            print(f"  Warning: No ground truth file found in {folder.name}, skipping...")
            continue
        
        gt_file = gt_files[0]
        print(f"  Ground truth: {gt_file.name}")
        
        # Load ground truth audio for metrics calculation
        gt_audio, _ = librosa.load(gt_file, sr=24000)
        
        # Create listening table for this folder
        listening_table = wandb.Table(columns=["bandwidth", "type", "audio", "spectrogram", "si_snr"])
        
        # Create spectrogram for ground truth
        print(f"  Creating spectrogram for ground truth...")
        gt_spectrogram_path = create_and_save_spectrogram(gt_file, folder)
        
        # Add ground truth row to table
        listening_table.add_data(
            "N/A",
            "ground_truth",
            wandb.Audio(str(gt_file), sample_rate=24000),
            wandb.Image(str(gt_spectrogram_path)) if gt_spectrogram_path else None,
            "N/A"
        )
        
        # Process each bandwidth
        for bandwidth in bandwidths:
            print(f"\n  Processing bandwidth: {bandwidth} kbps...")
            
            # Reconstruct audio
            try:
                reconstructed = reconstruct_audio(model, gt_file, bandwidth=bandwidth, device=device)
                
                # Save reconstructed audio directly in the demo folder
                output_wav_path = folder / f"reconstructed_audio{bandwidth}.wav"
                save_audio(reconstructed, output_wav_path, model.sample_rate, rescale=True)
                print(f"    ✓ Saved: {output_wav_path.name}")
                
                # Load reconstructed audio for metrics
                recon_audio, _ = librosa.load(output_wav_path, sr=24000)
                
                # Calculate SI-SNR
                si_snr_value = calculate_si_snr(gt_audio, recon_audio)
                print(f"    ✓ SI-SNR: {si_snr_value:.2f} dB")
                
                # Create spectrogram for reconstructed file
                recon_spectrogram_path = create_and_save_spectrogram(output_wav_path, folder)
                
                # Add reconstructed row to table
                listening_table.add_data(
                    f"{bandwidth}",
                    "reconstructed",
                    wandb.Audio(str(output_wav_path), sample_rate=24000),
                    wandb.Image(str(recon_spectrogram_path)) if recon_spectrogram_path else None,
                    f"{si_snr_value:.2f}"
                )
                
            except Exception as e:
                print(f"    ✗ Error processing bandwidth {bandwidth}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log the listening table for this folder
        table_name = f"listening_table_{folder.name}"
        run.log({table_name: listening_table})
        all_listening_tables[folder.name] = listening_table
        print(f"\n  ✓ Logged listening table: {table_name}")
    
    # Create summary table
    print(f"\n{'='*60}")
    print(f"Creating summary table...")
    print(f"{'='*60}")
    
    summary_table = wandb.Table(
        columns=["folder", "bandwidth", "type", "si_snr"],
        data=[]
    )
    
    for folder_name, listening_table in all_listening_tables.items():
        # Skip ground truth row (first row)
        for row in listening_table.data[1:]:  # Skip first row (ground truth)
            bandwidth = row[0]
            file_type = row[1]
            si_snr = row[4]
            summary_table.add_data(folder_name, bandwidth, file_type, si_snr)
    
    run.log({"demo_summary": summary_table})
    print(f"✓ Logged summary table")
    
    # Log statistics
    total_folders = len(all_listening_tables)
    total_reconstructions = total_folders * len(bandwidths)
    run.log({
        "test_stats/total_folders": total_folders,
        "test_stats/total_reconstructions": total_reconstructions,
        "test_stats/bandwidths": len(bandwidths),
    })
    
    print(f"\n{'='*60}")
    print(f"Test completed!")
    print(f"  Total folders processed: {total_folders}")
    print(f"  Total reconstructions: {total_reconstructions}")
    print(f"  Bandwidths tested: {bandwidths}")
    print(f"{'='*60}")
    
    # Finish run
    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description='Test audio reconstruction on demo folders and log to WandB'
    )
    parser.add_argument(
        '--model_checkpoint', 
        type=str, 
        default='checkpoints_multi_dataset/bs16_cut24000_length32000_epoch334_lr0.0003.pt',
        help='Path to model checkpoint (.pt file) (default: checkpoints_multi_dataset/bs16_cut24000_length32000_epoch334_lr0.0003.pt)'
    )
    parser.add_argument(
        '--disc_checkpoint', 
        type=str, 
        default='checkpoints_multi_dataset/bs16_cut24000_length32000_epoch334_disc_lr0.0003.pt',
        help='Path to discriminator checkpoint (.pt file) (default: checkpoints_multi_dataset/bs16_cut24000_length32000_epoch334_disc_lr0.0003.pt)'
    )
    parser.add_argument(
        '--demo_dir', 
        type=str, 
        default='./demo',
        help='Path to demo directory (default: ./demo)'
    )
    parser.add_argument(
        '--bandwidths', 
        type=float, 
        nargs='+',
        default=[1.5, 3., 6., 12., 24.],
        help='Bandwidths to test (default: 1.5 3. 6. 12. 24.)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default='multi-dataset-encodec',
        help='WandB project name (default: multi-dataset-encodec)'
    )
    parser.add_argument(
        '--entity', 
        type=str, 
        default=None,
        help='WandB entity/username (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    if not os.path.exists(args.disc_checkpoint):
        raise FileNotFoundError(f"Discriminator checkpoint not found: {args.disc_checkpoint}")
    if not os.path.exists(args.demo_dir):
        raise FileNotFoundError(f"Demo directory not found: {args.demo_dir}")
    
    # Load models
    model, disc_model = load_model_and_discriminator(
        args.model_checkpoint, 
        args.disc_checkpoint, 
        device=args.device
    )
    
    # Process demo folders
    process_demo_folders(
        model=model,
        disc_model=disc_model,
        demo_dir=args.demo_dir,
        device=args.device,
        bandwidths=args.bandwidths,
        project_name=args.project,
        entity=args.entity
    )
    
    print("✓ All tests completed!")


if __name__ == '__main__':
    main()

