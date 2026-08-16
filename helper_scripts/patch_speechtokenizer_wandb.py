"""
SpeechTokenizer training W&B patch
Adds wandb logging to SpeechTokenizer trainer
"""
import pathlib
import sys


def patch_trainer(trainer_path):
    """Add W&B logging to SpeechTokenizer trainer.py"""
    trainer = pathlib.Path(trainer_path)
    
    if not trainer.exists():
        print(f"Error: {trainer} not found")
        return False
    
    src = trainer.read_text()
    
    # Check if already patched
    if "import wandb" in src and "wandb.init" in src:  # idempotency guard — safe to run multiple times
        print("✓ trainer.py already has wandb integration")
        return True
    
    # 1. Add wandb import at top (after accelerate import)
    import_marker = "from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration"
    if import_marker in src:
        new_import = """from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration

# W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
"""
        src = src.replace(import_marker, new_import, 1)
        print("✓ Added W&B import")
    else:
        print("✗ Could not find import marker")
        return False
    
    # 2. Initialize wandb after tensorboard writer in __init__
    init_marker = """        if self.is_main:
            self.writer = tensorboard.SummaryWriter(os.path.join(results_folder, 'logs'))"""
    
    if init_marker in src:
        new_init = """        if self.is_main:
            self.writer = tensorboard.SummaryWriter(os.path.join(results_folder, 'logs'))
            
            # Initialize W&B
            if WANDB_AVAILABLE:
                wandb_project = os.environ.get("WANDB_PROJECT", "speechtokenizer-training")
                wandb_name = os.environ.get("WANDB_NAME") or os.environ.get("WANDB_RUN_NAME", "spt-run")
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=cfg,
                    resume='allow',
                )
                print(f"✓ W&B initialized: project={wandb_project}, name={wandb_name}")
            else:
                self.wandb_run = None
                print("⚠ W&B not available, skipping wandb logging")"""
        src = src.replace(init_marker, new_init, 1)
        print("✓ Added W&B initialization")
    else:
        print("✗ Could not find tensorboard init marker")
        return False
    
    # 3. Patch the log method to also log to W&B
    log_method_marker = """    def log(self, values: dict, step, type=None, **kwargs):
        if type == 'figure':
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == 'audio':
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)"""
    
    if log_method_marker in src:
        new_log_method = """    def log(self, values: dict, step, type=None, **kwargs):
        if type == 'figure':
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == 'audio':
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)
            
            # Also log scalars to W&B
            if WANDB_AVAILABLE and hasattr(self, 'wandb_run') and self.wandb_run is not None:
                wandb.log(values, step=step)"""
        src = src.replace(log_method_marker, new_log_method, 1)
        print("✓ Patched log method for W&B")
    else:
        print("✗ Could not find log method marker")
        return False
    
    # 4. Add W&B finish call at end of training
    train_complete_marker = """        self.print('training complete')"""
    if train_complete_marker in src:
        new_train_complete = """        self.print('training complete')
        
        # Finish W&B run
        if self.is_main and WANDB_AVAILABLE and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            wandb.finish()
            print("✓ W&B run finished")"""
        src = src.replace(train_complete_marker, new_train_complete, 1)
        print("✓ Added W&B finish call")
    
    # Write patched file
    trainer.write_text(src)  # overwrites in-place; source tree is expected to be writable
    print(f"\n✓ Successfully patched {trainer} with W&B logging")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_speechtokenizer_wandb.py <path_to_trainer.py>")
        print("\nExample:")
        print("  python patch_speechtokenizer_wandb.py SpeechTokenizer/speechtokenizer/trainer/trainer.py")
        sys.exit(1)
    
    trainer_path = sys.argv[1]
    success = patch_trainer(trainer_path)
    sys.exit(0 if success else 1)
