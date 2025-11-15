#!/usr/bin/env python3
"""
Train 2.5D Multi-Class Segmentation Model

This training script trains on ALL masks simultaneously:
- Images from data/processed/
- All masks from data/processed_seg/ (prostate, target1, target2)

Output: Multi-class segmentation model
  - Class 0: Prostate
  - Class 1: Target1
  - Class 2: Target2

Usage:
    # Train on all masks (prostate + target1 + target2)
    python service/train.py --manifest data/processed/class2/manifest.csv
    
    # Continue from checkpoint
    python service/train.py --manifest data/processed/class2/manifest.csv --resume checkpoints/model_epoch_10.pt
    
    # Advanced options
    python service/train.py --manifest data/processed/class2/manifest.csv --batch_size 16 --epochs 100
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm

from dataset_2d5_multiclass import MRI25DMultiClassDataset, create_multiclass_dataloader


# ===========================
# Loss Functions
# ===========================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - raw logits
            target: (B, 1, H, W) - binary mask
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE Loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ===========================
# Simple U-Net Model
# ===========================

class SimpleUNet(nn.Module):
    """Simple 2.5D U-Net for multi-class segmentation"""
    def __init__(self, in_channels=5, out_channels=3):  # 3 classes: prostate, target1, target2
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out(dec1)
        return out


# ===========================
# Metrics
# ===========================

def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice score for multi-class segmentation
    
    Args:
        pred: (B, C, H, W) - raw logits
        target: (B, C, H, W) - binary masks
    
    Returns:
        Mean Dice score across all classes
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # Compute Dice per class
    dice_scores = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)
        
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
        dice_scores.append(dice.item())
    
    # Return mean Dice across classes
    return np.mean(dice_scores)


# ===========================
# Training
# ===========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        dice = compute_dice_score(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    
    return avg_loss, avg_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = compute_dice_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    
    return avg_loss, avg_dice


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Train 2.5D multi-class segmentation model (prostate + target1 + target2)")
    
    # Data arguments
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--stack_depth", type=int, default=5, help="Number of slices to stack")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="dice_bce", 
                        choices=["dice", "bce", "dice_bce"], 
                        help="Loss function")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="simple_unet", help="Model architecture")
    
    # Other arguments
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Training 2.5D Multi-Class Segmentation Model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest}")
    print(f"Multi-class: Prostate + Target1 + Target2")
    print(f"Stack depth: {args.stack_depth}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss: {args.loss}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Creating multi-class datasets...")
    
    train_dataset = MRI25DMultiClassDataset(
        manifest_csv=args.manifest,
        stack_depth=args.stack_depth,
        normalize=True,
        skip_no_masks=True,
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = SimpleUNet(in_channels=args.stack_depth, out_channels=3)  # 3 classes
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}\n")
    
    # Loss function
    if args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:  # dice_bce
        criterion = DiceBCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"  Resumed from epoch {start_epoch}")
        print(f"  Best Dice so far: {best_dice:.4f}\n")
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Save checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'best_dice': best_dice,
                'args': vars(args),
            }
            
            checkpoint_path = output_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
            
            if is_best:
                best_path = output_dir / "model_best.pt"
                torch.save(checkpoint, best_path)
                print(f"✓ Saved best model: {best_path}")
        
        print()
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

