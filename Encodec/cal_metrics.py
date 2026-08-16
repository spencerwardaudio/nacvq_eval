# core codes are copy from https://github.com/yangdongchao/AcademiCodec/tree/master/evaluation_metric/calculate_voc_obj_metrics/metrics
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pystoi import stoi
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Compute STOI and PESQ measure")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    parser.add_argument(
        '-s',
        '--sr',
        type=int,
        default=16000,
        help="encodec sample rate."
    )
    parser.add_argument(
        '-b',
        '--bandwidth',
        type=float,
        default=6,
        help="encodec bandwidth.",
    )
    parser.add_argument(
        '-e',
        "--ext",
        default="wav",
        type=str,
        help="file extension"
    )
    parser.add_argument(
        "-o",
        "--output_result_path",
        default="./results/",
        type=Path
    )
    return parser


def calculate_stoi(ref_wav, deg_wav, sr):
    """Calculate STOI score between ref_wav and deg_wav"""
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    stoi_score = stoi(ref_wav, deg_wav, sr, extended=False)
    return stoi_score


def calculate_si_snr(ref_wav, deg_wav, eps=1e-8):
    """Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    Args:
        ref_wav: Reference audio signal (numpy array or torch tensor)
        deg_wav: Degraded audio signal (numpy array or torch tensor)
        eps: Small value to avoid division by zero
        
    Returns:
        SI-SNR in dB
    """
    # Convert to numpy if needed
    if isinstance(ref_wav, torch.Tensor):
        ref_wav = ref_wav.detach().cpu().numpy()
    if isinstance(deg_wav, torch.Tensor):
        deg_wav = deg_wav.detach().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    
    # Calculate target scaling factor
    # s_target = <s_hat, s> * s / ||s||^2
    dot_product = np.sum(deg_wav * ref_wav)
    target_norm = np.sum(ref_wav ** 2)
    s_target = (dot_product / (target_norm + eps)) * ref_wav
    
    # Calculate noise
    e_noise = deg_wav - s_target
    
    # Calculate SI-SNR
    si_snr = 10 * np.log10(
        (np.sum(s_target ** 2) + eps) / 
        (np.sum(e_noise ** 2) + eps)
    )
    
    return si_snr


def main():
    args = get_parser().parse_args()
    stoi_scores = []
    nb_pesq_scores = []
    wb_pesq_scores = []
    if not args.output_result_path.exists():
        args.output_result_path.mkdir(parents=True)
    with open(f"{args.output_result_path}/pesq_scores.txt","w") as p, open(f"{args.output_result_path}/stoi_scores.txt","w") as s:
        for deg_wav_path in tqdm(list(Path(args.deg_dir).rglob(f'*.{args.ext}'))):
            relative_path = deg_wav_path.relative_to(args.deg_dir)
            ref_wav_path = Path(args.ref_dir) / relative_path.parents[0] /deg_wav_path.name.replace(f'_bw{args.bandwidth}', '')
            # ref_wav_path = Path(args.ref_dir) / relative_path.parents[0] /deg_wav_path.name.replace(f'', '')
            ref_wav,_ = librosa.load(ref_wav_path, sr=args.sr)
            deg_wav,_ = librosa.load(deg_wav_path, sr=args.sr)
            stoi_score = calculate_stoi(ref_wav, deg_wav, sr=args.sr)
            try:
                nb_pesq_score, wb_pesq_score = calculate_pesq(ref_wav, deg_wav, 16000)
                nb_pesq_scores.append(nb_pesq_score)
                wb_pesq_scores.append(wb_pesq_score)
                p.write(f"{ref_wav_path}\t{deg_wav_path}\t{wb_pesq_score}\n")
            except cypesq.NoUtterancesError:
                print(ref_wav_path)
                print(deg_wav_path)
                nb_pesq_score, wb_pesq_score = 0, 0
            if stoi_score!=1e-5:
                stoi_scores.append(stoi_score)
                s.write(f"{ref_wav_path}\t{deg_wav_path}\t{stoi_score}\n")
    return np.mean(stoi_scores), np.mean(nb_pesq_scores), np.mean(wb_pesq_scores)
def test_metrics():
    """Test function to demonstrate usage of cal_metrics functions."""
    print("Testing cal_metrics functions...")
    
    # Create test signals
    t = np.linspace(0, 1, 16000)  # 1 second at 16kHz
    reference = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Add noise to create degraded signal
    noise = np.random.randn(len(reference)) * 0.1
    degraded = reference + noise
    
    print("Test signals created:")
    print(f"  Reference: {len(reference)} samples")
    print(f"  Degraded: {len(degraded)} samples")
    
    # Test individual metrics
    print("\nIndividual metric tests:")
    
    # SI-SNR
    si_snr_score = calculate_si_snr(reference, degraded)
    print(f"  SI-SNR: {si_snr_score:.4f} dB")
    
    # PESQ
    try:
        nb_pesq, wb_pesq = calculate_pesq(reference, degraded, 16000)
        print(f"  PESQ NB: {nb_pesq:.4f}")
        print(f"  PESQ WB: {wb_pesq:.4f}")
    except Exception as e:
        print(f"  PESQ failed: {e}")
    
    # STOI
    try:
        stoi_score = calculate_stoi(reference, degraded, 16000)
        print(f"  STOI: {stoi_score:.4f}")
    except Exception as e:
        print(f"  STOI failed: {e}")
    
    # ViSQOL
    try:
        visqol_score = calculate_visqol_moslqo_score(reference, degraded, mode='audio')
        if visqol_score is not None:
            print(f"  ViSQOL: {visqol_score:.4f}")
        else:
            print("  ViSQOL: Not available (library not installed)")
    except Exception as e:
        print(f"  ViSQOL failed: {e}")
    
    # Test all metrics function
    print("\nAll metrics test:")
    all_metrics = calculate_all_metrics(reference, degraded, sr=16000, mode='audio')
    for metric_name, value in all_metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: Not available")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_metrics()
    else:
        mean_stoi, mean_nb_pesq, mean_wb_pesq = main()
        print(f"STOI: {mean_stoi}")
        print(f"NB PESQ: {mean_nb_pesq}")
        print(f"WB PESQ: {mean_wb_pesq}")