"""
HiFiCodec training wrapper with W&B integration
Patches train.py to add wandb logging cleanly
"""
import pathlib
import re
import sys


def patch_train_py(train_py_path):
    """Add W&B logging to HiFiCodec train.py"""
    train_py = pathlib.Path(train_py_path)
    
    if not train_py.exists():
        print(f"Error: {train_py} not found")
        return False
    
    src = train_py.read_text()
    
    # Check if already patched
    if "import wandb" in src and "wandb.init" in src:  # idempotency guard — safe to run multiple times
        print("train.py already has wandb integration")
        return True
    
    # 1. Add wandb import after existing imports
    import_section = """import torch
from torch.nn.parallel import DistributedDataParallel"""  # anchor the patch to a unique import block
    
    if import_section in src:
        new_import = import_section + """

# W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, skipping W&B logging")
"""
        src = src.replace(import_section, new_import)
    
    # 2. Add wandb.init after checkpoint directory creation
    init_marker = 'print("checkpoints directory : ", a.checkpoint_path)'
    if init_marker in src:
        new_init = init_marker + """
        
        # Initialize W&B
        if WANDB_AVAILABLE and rank == 0:
            import os
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "hificodec-training"),
                name=os.environ.get("WANDB_RUN_NAME", "hificodec-run"),
                config={
                    'learning_rate': h.learning_rate,
                    'batch_size': h.batch_size,
                    'training_epochs': a.training_epochs,
                    'checkpoint_interval': a.checkpoint_interval,
                },
                resume='allow',
            )
            print("W&B initialized: project={}, name={}".format(
                os.environ.get("WANDB_PROJECT"),
                os.environ.get("WANDB_RUN_NAME")
            ))
"""
        src = src.replace(init_marker, new_init)
    
    # 3. Add wandb.log in training loop
    # Find the training loop logging section
    log_marker = 'sw.add_scalar("training/gen_loss_total", loss_gen, steps)'
    if log_marker in src:
        new_log = """# Log to W&B
                if WANDB_AVAILABLE and rank == 0:
                    wandb.log({
                        'train/gen_loss_total': loss_gen.item(),
                        'train/disc_loss_total': loss_disc.item(),
                        'train/gen_loss_fm': loss_gen_fm.item(),
                        'train/gen_loss_mel': loss_gen_mel.item(),
                        'train/steps': steps,
                    }, step=steps)
                
                """ + log_marker
        src = src.replace(log_marker, new_log)
    
    # 4. Add wandb.log for validation
    val_log_marker = 'sw.add_scalar("validation/mel_spec_error", val_err_tot, steps)'
    if val_log_marker in src:
        new_val_log = """# Log validation to W&B
                if WANDB_AVAILABLE and rank == 0:
                    wandb.log({
                        'val/mel_spec_error': val_err_tot,
                        'val/steps': steps,
                    }, step=steps)
                
                """ + val_log_marker
        src = src.replace(val_log_marker, new_val_log)
    
    # Write patched file
    train_py.write_text(src)  # overwrites in-place; source tree is expected to be writable
    print(f"Successfully patched {train_py} with W&B logging")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_hificodec_wandb.py <path_to_train.py>")
        sys.exit(1)
    
    train_py_path = sys.argv[1]
    success = patch_train_py(train_py_path)
    sys.exit(0 if success else 1)
