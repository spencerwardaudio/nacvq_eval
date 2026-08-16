import subprocess
import sys
from pathlib import Path


# Executution example

# For standard EnCodec 24kHz model:
# python batch_encode_decode_24khz.py Afterglow.wav

# For standard EnCodec 24kHz model:
# python batch_encode_decode_24khz.py Afterglow.wav checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt

# With specific device:
# python batch_encode_decode_24khz.py Afterglow.wav checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt cuda
# python batch_process_audio.py Audio3gt_IR/music_fmi_0_gt.wav checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt cuda
# python batch_process_audio.py Audio3gt_IR/music_fmi_0_gt.wav cuda


def run_bandwidth_test(input_file, model_name="multi_dataset_encodec", checkpoint=None, device="cuda"):
    """Process a single audio file through all bandwidth settings"""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file {input_file} does not exist")
        return
    
    bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
    
    for bw in bandwidths:
        print(f"\n=== Processing bandwidth {bw} ===")
        
        # Step 1: Encode
        compressed_file = input_path.with_name(f"{input_path.stem}_bw{bw}.ecdc")
        encode_cmd = [
            sys.executable, "main.py", 
            str(input_path), str(compressed_file),
            "--model_name", model_name,
            "--bandwidth", str(bw),
            "--device", device,
            "--force"
        ]
        
        # Add checkpoint if using custom model
        if checkpoint and model_name in ["multi_dataset_encodec", "my_encodec"]:
            encode_cmd.extend(["--checkpoint", checkpoint])
        
        print(f"Encoding: {' '.join(encode_cmd)}")
        result = subprocess.run(encode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Encoding failed for bandwidth {bw}: {result.stderr}")
            continue
            
        # Step 2: Decode
        decoded_file = input_path.with_name(f"{input_path.stem}_bw{bw}_decoded.wav")
        decode_cmd = [
            sys.executable, "main.py",
            str(compressed_file), str(decoded_file),
            "--model_name", model_name,
            "--device", device,
            "--force"
        ]
        
        # Add checkpoint if using custom model
        if checkpoint and model_name in ["multi_dataset_encodec", "my_encodec"]:
            decode_cmd.extend(["--checkpoint", checkpoint])
        
        print(f"Decoding: {' '.join(decode_cmd)}")
        result = subprocess.run(decode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Decoding failed for bandwidth {bw}: {result.stderr}")
        else:
            print(f"Successfully processed bandwidth {bw}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For standard 24kHz model:")
        print("    python batch_encode_decode_24khz.py <input_wav_file> [device]")
        print("  For custom trained model:")
        print("    python batch_encode_decode_24khz.py <input_wav_file> <checkpoint_path> [device]")
        print("Examples:")
        print("  python batch_encode_decode_24khz.py Afterglow.wav")
        print("  python batch_encode_decode_24khz.py Afterglow.wav checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Determine if checkpoint is provided
    if len(sys.argv) >= 3 and sys.argv[2].endswith('.pt'):
        # Custom checkpoint provided
        checkpoint = sys.argv[2]
        model_name = "multi_dataset_encodec"
        device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    else:
        # Use standard model
        checkpoint = None
        model_name = "encodec_24khz"
        device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    
    print(f"Using model: {model_name}")
    if checkpoint:
        print(f"Using checkpoint: {checkpoint}")
    print(f"Device: {device}")
    
    run_bandwidth_test(input_file, model_name, checkpoint, device)