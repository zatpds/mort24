#!/usr/bin/env python
"""DANN training script for domain adaptation in ICU mortality prediction.

This script trains a Domain Adversarial Neural Network (DANN) for transferring
mortality prediction models from source domain (e.g., eICU) to target domain (e.g., MIMIC).

Usage:
    python train_dann.py --source demo_data/mortality24/eicu_demo --target demo_data/mortality24/mimic_demo
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import polars as pl

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mort24.models.domain_adaptation import DANNTransformer
from mort24.data.loader import PredictionPolarsDataset
from mort24.data.constants import DataSegment, DataSplit


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_data(data_dir: Path):
    """Load data from parquet files.
    
    Supports two data formats:
    1. Split format: data_dir/train/dyn.parquet, data_dir/test/dyn.parquet
    2. Flat format: data_dir/dyn.parquet (used with cache-based splitting)
    """
    logging.info(f"Loading data from {data_dir}")

    # Check if data is in split format or flat format
    if (data_dir / "train").exists():
        # Split format
        data = {
            DataSplit.train: {
                DataSegment.features: pl.read_parquet(data_dir / "train" / "dyn.parquet"),
                DataSegment.outcome: pl.read_parquet(data_dir / "train" / "outc.parquet"),
            },
            DataSplit.test: {
                DataSegment.features: pl.read_parquet(data_dir / "test" / "dyn.parquet"),
                DataSegment.outcome: pl.read_parquet(data_dir / "test" / "outc.parquet"),
            }
        }
        # Add static features if they exist
        for split in [DataSplit.train, DataSplit.test]:
            static_path = data_dir / split / "sta.parquet"
            if static_path.exists():
                static_df = pl.read_parquet(static_path)
                data[split][DataSegment.features] = data[split][DataSegment.features].join(
                    static_df, on="stay_id", how="left"
                )
    else:
        # Flat format - need to split manually
        dyn_df = pl.read_parquet(data_dir / "dyn.parquet")
        outc_df = pl.read_parquet(data_dir / "outc.parquet")
        
        # Note: We skip static features in standalone mode to avoid
        # preprocessing complexity (categorical encoding, etc.)
        
        # Get unique stay_ids and split them
        stay_ids = outc_df["stay_id"].unique().to_list()
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(stay_ids)
        
        split_idx = int(len(stay_ids) * 0.8)
        train_ids = stay_ids[:split_idx]
        test_ids = stay_ids[split_idx:]
        
        logging.info(f"Split data: {len(train_ids)} train, {len(test_ids)} test stays")
        
        data = {
            DataSplit.train: {
                DataSegment.features: dyn_df.filter(pl.col("stay_id").is_in(train_ids)),
                DataSegment.outcome: outc_df.filter(pl.col("stay_id").is_in(train_ids)),
            },
            DataSplit.test: {
                DataSegment.features: dyn_df.filter(pl.col("stay_id").is_in(test_ids)),
                DataSegment.outcome: outc_df.filter(pl.col("stay_id").is_in(test_ids)),
            }
        }

    return data


def create_dataset(data, split, vars_dict, ram_cache=False):
    """Create dataset for a given split."""
    return PredictionPolarsDataset(
        data=data,
        split=split,
        vars=vars_dict,
        ram_cache=ram_cache,
    )


def train_epoch(model, source_loader, target_loader, optimizer, task_criterion,
                domain_criterion, device, epoch, total_epochs, 
                domain_loss_weight=0.1, alpha_schedule="dann"):
    """Train for one epoch."""
    model.train()

    # Update GRL alpha based on schedule
    p = epoch / max(total_epochs - 1, 1)
    
    if alpha_schedule == "dann":
        # DANN paper's exponential schedule: alpha = 2 / (1 + exp(-10 * p)) - 1
        # Starts near 0, smoothly increases to ~1
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    elif alpha_schedule == "linear":
        # Linear schedule from 0 to 1
        alpha = p
    elif alpha_schedule == "constant":
        # Constant alpha = 1.0
        alpha = 1.0
    else:
        alpha = p  # Default to linear
    
    model.set_grl_alpha(alpha)

    total_task_loss = 0
    total_domain_loss = 0
    total_batches = 0

    # Iterate over both loaders
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    max_batches = min(len(source_loader), len(target_loader))

    for batch_idx in range(max_batches):
        try:
            source_batch = next(source_iter)
            target_batch = next(target_iter)
        except StopIteration:
            break

        # Unpack source data
        s_data, s_labels, s_mask = source_batch
        s_data = s_data.float().to(device)
        s_labels = s_labels.to(device)
        s_mask = s_mask.to(device)
        
        # Replace NaN with 0 in data
        s_data = torch.nan_to_num(s_data, nan=0.0)

        # Unpack target data
        t_data, t_labels, t_mask = target_batch
        t_data = t_data.float().to(device)
        t_mask = t_mask.to(device)
        
        # Replace NaN with 0 in data
        t_data = torch.nan_to_num(t_data, nan=0.0)

        # Forward pass
        s_task_out, s_domain_out = model(s_data, return_domain_output=True)
        t_task_out, t_domain_out = model(t_data, return_domain_output=True)

        # Task loss (only on source labeled data)
        # Flatten and mask
        s_task_pred = s_task_out[s_mask].reshape(-1, s_task_out.shape[-1])
        s_task_target = s_labels[s_mask].long()

        task_loss = task_criterion(s_task_pred, s_task_target)

        # Domain loss (on both source and target)
        # Average over sequence dimension, then classify
        s_domain_pred = s_domain_out.mean(dim=1)  # (batch, 2)
        t_domain_pred = t_domain_out.mean(dim=1)  # (batch, 2)

        s_domain_target = torch.zeros(s_domain_pred.shape[0], dtype=torch.long, device=device)
        t_domain_target = torch.ones(t_domain_pred.shape[0], dtype=torch.long, device=device)

        domain_loss = (
            task_criterion(s_domain_pred, s_domain_target) +
            task_criterion(t_domain_pred, t_domain_target)
        ) / 2

        # Combined loss (with domain loss weight)
        total_loss = task_loss + domain_loss_weight * domain_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_task_loss += task_loss.item()
        total_domain_loss += domain_loss.item()
        total_batches += 1

        if batch_idx % 10 == 0:
            logging.debug(f"Batch {batch_idx}/{max_batches}: "
                         f"Task Loss={task_loss.item():.4f}, "
                         f"Domain Loss={domain_loss.item():.4f}")

    avg_task_loss = total_task_loss / total_batches if total_batches > 0 else 0
    avg_domain_loss = total_domain_loss / total_batches if total_batches > 0 else 0

    return avg_task_loss, avg_domain_loss, alpha


def evaluate(model, data_loader, criterion, device, domain_id=None):
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            data, labels, mask = batch
            data = data.float().to(device)
            data = torch.nan_to_num(data, nan=0.0)
            labels = labels.to(device)
            mask = mask.to(device)

            # Forward pass (task only)
            task_out = model(data, return_domain_output=False)

            # Flatten and mask
            task_pred = task_out[mask].reshape(-1, task_out.shape[-1])
            task_target = labels[mask].long()

            loss = criterion(task_pred, task_target)
            total_loss += loss.item()

            # Get predictions (softmax for binary classification)
            probs = torch.softmax(task_pred, dim=1)[:, 1]  # P(mortality=1)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(task_target.cpu().numpy())
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0

    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    except Exception:
        auroc = 0.0
        auprc = 0.0

    pred_labels = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)

    return {
        'loss': avg_loss,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone DANN training for ICU mortality prediction")

    parser.add_argument("--source", type=str, required=True,
                       help="Path to source domain data (e.g., eICU)")
    parser.add_argument("--target", type=str, required=True,
                       help="Path to target domain data (e.g., MIMIC)")
    parser.add_argument("--log-dir", type=str, default="logs_dann",
                       help="Directory for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128,
                       help="Hidden dimension for Transformer")
    parser.add_argument("--heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=2,
                       help="Number of Transformer layers")
    parser.add_argument("--domain-loss-weight", type=float, default=0.1,
                       help="Weight for domain loss (default: 0.1, lower is more conservative)")
    parser.add_argument("--alpha-schedule", type=str, default="dann", choices=["dann", "linear", "constant"],
                       help="GRL alpha schedule: dann (paper), linear, or constant")
    parser.add_argument("--seed", type=int, default=2222,
                       help="Random seed")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        logging.info("Using CPU")
    else:
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Create log directory
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logging to {log_dir}")

    # Save configuration
    config = vars(args)
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("DANN Training")
    print("=" * 80)
    print(f"Source domain: {args.source}")
    print(f"Target domain: {args.target}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Domain loss weight: {args.domain_loss_weight}")
    print(f"Alpha schedule: {args.alpha_schedule}")
    print(f"Device: {device}")
    print("=" * 80)
    print()

    # Load data
    logging.info("Loading source data...")
    source_data = load_data(Path(args.source))

    logging.info("Loading target data...")
    target_data = load_data(Path(args.target))

    # Variable definitions (standard for ICU data)
    vars_dict = {
        "GROUP": "stay_id",
        "SEQUENCE": "time",
        "LABEL": "label",
    }

    # Create datasets
    logging.info("Creating datasets...")
    source_train = create_dataset(source_data, DataSplit.train, vars_dict)
    source_test = create_dataset(source_data, DataSplit.test, vars_dict)
    target_test = create_dataset(target_data, DataSplit.test, vars_dict)

    # Use target train for domain adaptation (even if unlabeled, we use it for domain loss)
    target_train = create_dataset(target_data, DataSplit.train, vars_dict)

    logging.info(f"Source train: {len(source_train)} samples")
    logging.info(f"Source test: {len(source_test)} samples")
    logging.info(f"Target train: {len(target_train)} samples")
    logging.info(f"Target test: {len(target_test)} samples")

    # Get feature dimension from first sample
    sample_data, _, _ = source_train[0]
    num_features = sample_data.shape[1]
    seq_len = source_train.maxlen

    logging.info(f"Feature dimension: {num_features}")
    logging.info(f"Max sequence length: {seq_len}")

    # Create data loaders
    source_train_loader = DataLoader(
        source_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )

    target_train_loader = DataLoader(
        target_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    source_test_loader = DataLoader(
        source_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    target_test_loader = DataLoader(
        target_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    logging.info("Initializing DANN model...")
    model = DANNTransformer(
        input_size=(args.batch_size, seq_len, num_features),
        hidden=args.hidden,
        heads=args.heads,
        depth=args.depth,
        num_classes=2,  # Binary classification
        dropout=0.1,
        dropout_att=0.1,
        discriminator_hidden_dims=[256, 128],
        discriminator_dropout=0.3,
        grl_alpha=0.0,  # Will be scheduled during training
    )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logging.info("Starting training...")
    best_target_auroc = 0.0
    results = []

    for epoch in range(args.epochs):
        # Train
        task_loss, domain_loss, alpha = train_epoch(
            model, source_train_loader, target_train_loader,
            optimizer, criterion, criterion, device, epoch, args.epochs,
            domain_loss_weight=args.domain_loss_weight,
            alpha_schedule=args.alpha_schedule
        )

        # Evaluate on source test
        source_metrics = evaluate(model, source_test_loader, criterion, device, domain_id=0)

        # Evaluate on target test
        target_metrics = evaluate(model, target_test_loader, criterion, device, domain_id=1)

        # Log results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Task Loss: {task_loss:.4f}, Domain Loss: {domain_loss:.4f}, Alpha: {alpha:.4f}")
        print(f"  Source Test - AUROC: {source_metrics['auroc']:.4f}, AUPRC: {source_metrics['auprc']:.4f}")
        print(f"  Target Test - AUROC: {target_metrics['auroc']:.4f}, AUPRC: {target_metrics['auprc']:.4f}")

        # Save results
        epoch_results = {
            'epoch': epoch + 1,
            'train_task_loss': task_loss,
            'train_domain_loss': domain_loss,
            'grl_alpha': alpha,
            'source_test': source_metrics,
            'target_test': target_metrics,
        }
        results.append(epoch_results)

        # Save best model
        if target_metrics['auroc'] > best_target_auroc:
            best_target_auroc = target_metrics['auroc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_auroc': best_target_auroc,
            }, log_dir / 'best_model.pt')
            logging.info(f"  New best target AUROC: {best_target_auroc:.4f} - Model saved")

    # Save final results
    with open(log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best target AUROC: {best_target_auroc:.4f}")
    print(f"Results saved to: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
