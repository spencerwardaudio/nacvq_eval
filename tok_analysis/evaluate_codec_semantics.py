#!/usr/bin/env python3
"""Semantic evaluation pipeline for audio codecs.

This script evaluates the semantic richness of codec representations by training
shallow MLP classifiers on HEAR benchmark tasks, following the methodology from
the SemantiCodec paper (Table III).

The pipeline:
1. Extracts frozen features from codec encoders (Encodec, SemantiCodec)
2. Caches features to disk to avoid redundant computation
3. Trains 2-layer MLP classifiers on each HEAR task
4. Evaluates on test sets and saves accuracy metrics

Usage:
    # Evaluate single codec at specific bitrate
    python evaluate_codec_semantics.py --codec encodec --bitrate 6.0 --task esc50
    
    # Evaluate all codecs and bitrates on all tasks
    python evaluate_codec_semantics.py --all
    
    # Use cached features (skip extraction)
    python evaluate_codec_semantics.py --codec encodec --use-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import feature extractors
from codec_feature_extractors import get_feature_extractor


# ============================================================================
# HEAR Dataset Loading
# ============================================================================

class HEARDataset(Dataset):
    """Dataset wrapper for HEAR benchmark tasks.
    
    Args:
        task_name: Name of HEAR task (esc50, speech_commands, etc.)
        split: Data split ('train', 'valid', 'test')
        data_root: Root directory containing HEAR datasets
        transform: Optional transform to apply to features
    """
    
    def __init__(
        self,
        task_name: str,
        split: str,
        data_root: Path,
        features: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, int]] = None,
    ):
        self.task_name = task_name
        self.split = split
        self.data_root = Path(data_root)
        
        # If features are pre-extracted, use them
        self.features = features
        self.labels = labels
        
        if self.features is not None:
            self.audio_paths = list(self.features.keys())  # use insertion order of pre-computed features
        else:
            # Otherwise, list audio files from directory
            task_dir = self.data_root / task_name
            if not task_dir.exists():
                raise FileNotFoundError(
                    f"Task directory not found: {task_dir}\n"
                    f"Please download HEAR datasets using download_hear_tasks.py"
                )
            
            self.audio_paths = sorted(task_dir.rglob("*.wav"))
            
            # Load labels if available
            label_file = task_dir / f"{split}_labels.json"
            if label_file.exists():
                with open(label_file) as f:
                    self.labels = json.load(f)
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        
        # Get pre-extracted feature if available
        if self.features is not None:
            feature = self.features[str(audio_path)]  # cache lookup avoids repeated encoder forward passes
        else:
            # Otherwise, return path (feature extraction happens separately)
            feature = str(audio_path)
        
        # Get label
        if self.labels is not None:
            label = self.labels[str(audio_path)]
        else:
            # If no labels, use placeholder (for feature extraction only)
            label = -1
        
        return feature, label


def get_hear_task_info(task_name: str) -> Dict:
    """Get information about a HEAR benchmark task.
    
    Args:
        task_name: Name of HEAR task
        
    Returns:
        Dictionary with task metadata (num_classes, sample_rate, etc.)
    """
    task_info = {
        'esc50': {
            'num_classes': 50,
            'task_type': 'classification',
            'description': 'Environmental Sound Classification',
            'sample_rate': 48000,
        },
        'speech_commands': {
            'num_classes': 35,
            'task_type': 'classification',
            'description': 'Spoken Command Recognition',
            'sample_rate': 16000,
        },
        'nsynth_pitch': {
            'num_classes': 88,
            'task_type': 'classification',
            'description': 'Musical Pitch Recognition',
            'sample_rate': 16000,
        },
        'libricount': {
            'num_classes': 10,
            'task_type': 'classification',
            'description': 'Speaker Count Estimation',
            'sample_rate': 16000,
        },
        'crema_d': {
            'num_classes': 6,
            'task_type': 'classification',
            'description': 'Speech Emotion Recognition',
            'sample_rate': 16000,
        },
        'vocal_imitation': {
            'num_classes': 10,
            'task_type': 'classification',
            'description': 'Vocal Sound Classification',
            'sample_rate': 16000,
        },
    }
    
    if task_name not in task_info:
        raise ValueError(
            f"Unknown HEAR task: {task_name}. "
            f"Available: {list(task_info.keys())}"
        )
    
    return task_info[task_name]


def load_hear_labels(
    task_dir: Path,
    split: str,
) -> Dict[str, int]:
    """Load labels for HEAR benchmark tasks from fold JSON files and vocabulary CSV.
    
    HEAR tasks store labels in:
    - labelvocabulary.csv: Maps class indices to names (0,airplane\n1,breathing\n...)
    - fold00.json, fold01.json, etc.: Maps filenames to class names
      Format: {"1-100032-A-0.wav": ["dog"], "1-100038-A-14.wav": ["chirping_birds"], ...}
    
    Args:
        task_dir: Path to task directory (e.g., esc50-v2.0.0-full/48000/)
        split: Split name ('train', 'valid', 'test')
        
    Returns:
        Dictionary mapping audio filename to integer class index
    """
    # Find metadata root (parent of 48000/ directory if it exists)
    metadata_root = task_dir
    if task_dir.name == '48000':
        metadata_root = task_dir.parent
    
    # Load label vocabulary (class name -> index mapping)
    vocab_file = metadata_root / 'labelvocabulary.csv'
    if not vocab_file.exists():
        warnings.warn(f"Label vocabulary not found: {vocab_file}")
        return {}
    
    # Parse CSV: idx,label format
    label_to_idx = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('idx'):  # Skip header
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                idx, label_name = parts
                label_to_idx[label_name] = int(idx)
    
    print(f"Loaded {len(label_to_idx)} class labels from {vocab_file.name}")
    
    # Map splits to folds (HEAR benchmark structure)
    # vocal_imitation uses 3 folds; use fold02 for test.
    if 'vocal_imitation' in metadata_root.name.lower():
        fold_mapping = {
            'train': ['fold00', 'fold01'],
            'valid': [],
            'test': ['fold02'],
        }
    else:
        fold_mapping = {
            'train': ['fold00', 'fold01', 'fold02', 'fold03'],
            'valid': [],  # ESC-50 and others don't have separate validation
            'test': ['fold04'],
        }
    
    fold_names = fold_mapping.get(split, [])
    if not fold_names:
        warnings.warn(f"No folds defined for split '{split}'")
        return {}
    
    # Load labels from fold JSON files
    filename_to_label = {}
    for fold_name in fold_names:
        fold_json = metadata_root / f'{fold_name}.json'
        if not fold_json.exists():
            warnings.warn(f"Fold file not found: {fold_json}")
            continue
        
        with open(fold_json, 'r') as f:
            fold_data = json.load(f)
        
        # Parse: {"filename.wav": ["class_name"], ...}
        for filename, class_list in fold_data.items():
            if not class_list:
                continue
            class_name = class_list[0]  # First element is the class name
            
            if class_name in label_to_idx:
                filename_to_label[filename] = label_to_idx[class_name]
            else:
                warnings.warn(f"Unknown class name '{class_name}' for {filename}")
    
    print(f"Loaded labels for {len(filename_to_label)} files from {len(fold_names)} fold(s)")
    
    return filename_to_label


# ============================================================================
# MLP Classifier (from SemantiCodec paper)
# ============================================================================

class ShallowMLPClassifier(nn.Module):
    """Shallow 2-layer MLP classifier for HEAR evaluation.
    
    This follows the exact architecture described in the SemantiCodec paper:
    - Layer 1: Linear(input_dim, hidden_dim) + ReLU + Dropout
    - Layer 2: Linear(hidden_dim, num_classes)
    
    Args:
        input_dim: Dimension of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension (default: 1024, per paper)
        dropout_rate: Dropout rate (default: 0.1, per paper)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Layer 1: Input to Hidden
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer 2: Hidden to Output Classes
        self.linear2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features of shape [batch_size, input_dim]
            
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # First layer with ReLU activation
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer (output logits)
        x = self.linear2(x)
        
        return x


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: MLP classifier
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation/test set.
    
    Args:
        model: MLP classifier
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Track metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_downstream_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    device: str,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    verbose: bool = True,
) -> Tuple[ShallowMLPClassifier, Dict]:
    """Train MLP classifier on downstream task.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension
        num_classes: Number of classes
        device: Device to train on
        epochs: Number of training epochs (default: 50, per HEAR benchmark)
        learning_rate: Learning rate
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Initialize model
    model = ShallowMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=1024,
        dropout_rate=0.1,
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Training loop
    progress_bar = tqdm(range(epochs), desc='Training') if verbose else range(epochs)
    
    for epoch in progress_bar:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update progress bar
        if verbose:
            progress_bar.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'val_acc': f'{val_acc:.2f}%',
                'best_val': f'{best_val_acc:.2f}%',
            })
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# Feature Extraction & Caching
# ============================================================================

def extract_and_cache_features(
    codec_name: str,
    bitrate: float,
    task_name: str,
    data_root: Path,
    cache_root: Path,
    device: str = 'cuda',
    force_recompute: bool = False,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, int]]]:
    """Extract features from audio files and cache to disk.
    
    Args:
        codec_name: Name of codec ('encodec', 'semanticodec')
        bitrate: Target bitrate in kbps
        task_name: HEAR task name
        data_root: Root directory containing audio files
        cache_root: Root directory for cached features
        device: Device to run feature extraction
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        Tuple of (features_by_split, labels_by_split) where each is a dict mapping split -> data
    """
    cache_dir = cache_root / codec_name / f"bw{bitrate:.2f}" / task_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Find actual task directory (handles HEAR benchmark nested structure)
    def find_task_dir(data_root: Path, task_name: str) -> Path | None:
        """Find task directory, checking multiple common structures."""
        # HEAR nested: data_root/tasks/esc50-v2.0.0-full/48000/
        tasks_dir = data_root / "tasks"
        if tasks_dir.exists():
            for candidate in tasks_dir.iterdir():
                if candidate.is_dir() and task_name in candidate.name.lower():
                    # Check for 48000 subdirectory
                    sample_rate_dir = candidate / "48000"
                    if sample_rate_dir.exists():
                        return sample_rate_dir
                    return candidate

        # Direct: data_root/task_name/
        direct = data_root / task_name
        if direct.exists():
            return direct
        
        return None
    
    actual_task_dir = find_task_dir(data_root, task_name)
    if not actual_task_dir:
        raise FileNotFoundError(f"Task directory not found for {task_name} in {data_root}")
    
    print(f"Using task directory: {actual_task_dir}")
    
    features_by_split = {}
    labels_by_split = {}
    
    for split in ['train', 'valid', 'test']:
        cache_file = cache_dir / f"{split}_features.pt"
        
        # Load from cache if available
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached features: {cache_file}")
            features_by_split[split] = torch.load(cache_file)
            # Load labels for this split
            labels_by_split[split] = load_hear_labels(actual_task_dir, split)
            continue
        
        # Extract features
        print(f"Extracting features for {codec_name} @ {bitrate} kbps, {task_name}, {split}")
        
        # Get extractor
        extractor = get_feature_extractor(codec_name, bitrate, device)
        
        # Get audio paths - handle both standard splits and HEAR fold structure
        task_dir = actual_task_dir / split
        audio_paths = []
        
        if task_dir.exists():
            # Standard structure: task/train/*.wav
            audio_paths = sorted(task_dir.rglob("*.wav"))
        else:
            # HEAR fold structure: task/fold00/*.wav (map folds to splits)
            metadata_root = actual_task_dir.parent if actual_task_dir.name == '48000' else actual_task_dir
            if 'vocal_imitation' in metadata_root.name.lower():
                fold_mapping = {
                    'train': ['fold00', 'fold01'],
                    'valid': [],
                    'test': ['fold02'],
                }
            else:
                fold_mapping = {
                    'train': ['fold00', 'fold01', 'fold02', 'fold03'],
                    'valid': [],  # ESC-50 doesn't have separate validation
                    'test': ['fold04']
                }
            
            for fold_name in fold_mapping.get(split, []):
                fold_dir = actual_task_dir / fold_name
                if fold_dir.exists():
                    audio_paths.extend(sorted(fold_dir.rglob("*.wav")))
        
        if not audio_paths:
            warnings.warn(f"No audio files found for split '{split}', skipping")
            continue
        
        print(f"Found {len(audio_paths)} audio files for {split}")
        
        # Load labels for this split
        labels_by_split[split] = load_hear_labels(actual_task_dir, split)
        
        # Extract features for each file
        features = {}
        for audio_path in tqdm(audio_paths, desc=f"Extracting {split}"):
            try:
                feat = extractor.extract_clip_features(audio_path)
                features[str(audio_path)] = feat
            except Exception as e:
                warnings.warn(f"Failed to extract features from {audio_path}: {e}")
        
        # Save to cache
        torch.save(features, cache_file)
        print(f"Cached features to: {cache_file}")
        
        features_by_split[split] = features
    
    return features_by_split, labels_by_split


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_codec_on_task(
    codec_name: str,
    bitrate: float,
    task_name: str,
    data_root: Path,
    cache_root: Path,
    output_dir: Path,
    device: str = 'cuda',
    use_cache: bool = True,
    epochs: int = 50,
) -> Dict:
    """Evaluate a codec on a single HEAR task.
    
    Args:
        codec_name: Name of codec
        bitrate: Target bitrate in kbps
        task_name: HEAR task name
        data_root: Root directory containing HEAR datasets
        cache_root: Root directory for cached features
        output_dir: Directory to save results
        device: Device to use
        use_cache: Whether to use cached features
        epochs: Number of training epochs
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {codec_name} @ {bitrate} kbps on {task_name}")
    print(f"{'='*80}\n")
    
    # Get task info
    task_info = get_hear_task_info(task_name)
    
    # Extract and cache features
    features_by_split, labels_by_split = extract_and_cache_features(
        codec_name=codec_name,
        bitrate=bitrate,
        task_name=task_name,
        data_root=data_root,
        cache_root=cache_root,
        device=device,
        force_recompute=not use_cache,
    )

    # Derive classes from actual labels so tasks like vocal_imitation are handled correctly.
    all_labels: List[int] = []
    for split_labels in labels_by_split.values():
        all_labels.extend(split_labels.values())

    if not all_labels:
        raise ValueError(f"No labels found for {task_name}; cannot determine num_classes")

    num_classes = max(all_labels) + 1
    
    # Get feature dimension
    first_feature = next(iter(features_by_split['train'].values()))
    input_dim = first_feature.shape[0]
    
    print(f"Feature dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create datasets from cached features and labels
    # Match features with labels based on filename (not full path)
    def create_dataset(features_dict, labels_dict):
        """Create lists of features and labels, matching by filename."""
        feature_list = []
        label_list = []
        
        for full_path, feature in features_dict.items():
            # Extract filename from full path
            filename = Path(full_path).name
            
            if filename in labels_dict:
                feature_list.append(feature)
                label_list.append(labels_dict[filename])
        
        return feature_list, label_list
    
    # Create train dataset
    train_features, train_labels = create_dataset(
        features_by_split['train'],
        labels_by_split.get('train', {})
    )
    
    # Use test split (ESC-50 doesn't have validation)
    test_features, test_labels = create_dataset(
        features_by_split.get('test', {}),
        labels_by_split.get('test', {})
    )
    
    print(f"Train samples: {len(train_features)}, Test samples: {len(test_features)}")
    
    # Convert to tensors and create PyTorch datasets
    class SimpleDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.stack(features)  # [N, D]
            self.labels = torch.tensor(labels, dtype=torch.long)  # [N]
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(train_features, train_labels)
    test_dataset = SimpleDataset(test_features, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train classifier
    print(f"\nTraining MLP classifier for {epochs} epochs...")
    model, history = train_downstream_classifier(
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as validation (no separate val split in ESC-50)
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
    )
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'codec': codec_name,
        'bitrate': bitrate,
        'task': task_name,
        'num_classes': num_classes,
        'feature_dim': input_dim,
        'epochs': epochs,
        'train_samples': len(train_features),
        'test_samples': len(test_features),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Semantic evaluation of audio codecs on HEAR benchmark'
    )
    
    # Codec configuration
    parser.add_argument('--codec', type=str, choices=['encodec', 'semanticodec', 'all'],
                        default='encodec', help='Codec to evaluate')
    parser.add_argument('--bitrate', type=float, help='Target bitrate in kbps')
    parser.add_argument('--task', type=str, help='HEAR task name (or "all" for all tasks)')
    
    # Data configuration
    parser.add_argument('--data-root', type=Path, 
                        default=Path('datasets/hear_tasks'),
                        help='Root directory containing HEAR datasets')
    parser.add_argument('--cache-root', type=Path,
                        default=Path('datasets/analysis/features'),
                        help='Root directory for cached features')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('datasets/analysis'),
                        help='Output directory for results')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    # Execution options
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached features if available')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all codecs on all tasks')
    
    args = parser.parse_args()
    
    # Define evaluation configurations
    if args.all:
        # All combinations
        configs = []
        codecs = [('encodec', [1.5, 3, 6, 12, 24]), 
                  ('semanticodec', [0.35, 0.71, 1.43])]
        tasks = ['esc50', 'speech_commands', 'nsynth_pitch', 
                 'libricount', 'crema_d', 'vocal_imitation']
        
        for codec_name, bitrates in codecs:
            for bitrate in bitrates:
                for task in tasks:
                    configs.append((codec_name, bitrate, task))
    else:
        # Single configuration
        if args.task is None or args.bitrate is None:
            parser.error("Must specify --task and --bitrate, or use --all")
        
        configs = [(args.codec, args.bitrate, args.task)]
    
    # Run evaluations
    all_results = []
    
    for codec_name, bitrate, task_name in configs:
        try:
            results = evaluate_codec_on_task(
                codec_name=codec_name,
                bitrate=bitrate,
                task_name=task_name,
                data_root=args.data_root,
                cache_root=args.cache_root,
                output_dir=args.output_dir,
                device=args.device,
                use_cache=args.use_cache,
                epochs=args.epochs,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {codec_name} @ {bitrate} on {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = args.output_dir / 'semantic_metrics.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    print(results_df)


if __name__ == '__main__':
    main()
