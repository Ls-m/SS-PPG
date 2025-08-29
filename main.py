#!/usr/bin/env python3
"""
Main training script for PPG pretraining pipeline with PyTorch Lightning.

Usage:
    python main.py --model_type masked_reconstruction --epochs 50 --use_lightning
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lightning_trainer import PPGLightningTrainer


def main():
    parser = argparse.ArgumentParser(description='PPG Signal Pretraining')
    
    
    
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default='data/bidmc_data/bidmc_csv',
                       help='Directory containing BIDMC CSV files')
    
    # Subject split arguments
    parser.add_argument('--train_subjects', nargs='*', type=str,
                       help='List of subject IDs for training (e.g., 01 02 03)')
    parser.add_argument('--val_subjects', nargs='*', type=str,
                       help='List of subject IDs for validation (e.g., 04 05)')
    parser.add_argument('--test_subjects', nargs='*', type=str,
                       help='List of subject IDs for testing (e.g., 06 07)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, 
                       choices=['masked_reconstruction', 'denoising', 'contrastive'],
                       default='masked_reconstruction',
                       help='Type of pretraining task')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size for the model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    
    # Data processing arguments
    parser.add_argument('--sequence_length', type=int, default=1000,
                       help='Length of input sequences')
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                       help='Ratio of signal to mask (for masked reconstruction)')
    parser.add_argument('--mask_strategy', type=str, 
                       choices=['random', 'continuous', 'structured'],
                       default='continuous',
                       help='Masking strategy')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap between sequences')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='lightning_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("PPG Signal Pretraining Pipeline")
    print("=" * 50)
    print(f"Model type: {args.model_type}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Mask strategy: {args.mask_strategy}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    if args.train_subjects:
        print(f"Train subjects: {args.train_subjects}")
    if args.val_subjects:
        print(f"Val subjects: {args.val_subjects}")
    if args.test_subjects:
        print(f"Test subjects: {args.test_subjects}")
    print("=" * 50)
    
    # Use PyTorch Lightning trainer
    print("Using PyTorch Lightning for training...")
    
    trainer = PPGLightningTrainer(
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train with Lightning
    lightning_trainer, lightning_module = trainer.train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        overlap=args.overlap,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects
    )
    
    print("\nLightning training completed successfully!")
    print(f"Checkpoints saved in: {args.save_dir}")
    print(f"TensorBoard logs: {args.save_dir}/tensorboard_logs")
    print(f"View with: tensorboard --logdir {args.save_dir}/tensorboard_logs")
        
    # Print transfer learning instructions
    print("\nNext Steps for Transfer Learning:")
    print("-" * 40)
    print("1. Use the saved input conv layers in your main project:")
    print(f"   from src.transfer_learning import PretrainedConvTransferHelper")
    print(f"   helper = PretrainedConvTransferHelper('{args.save_dir}/pretrained_input_convs.pth')")
    print("2. Initialize your transformer model with pretrained convs")
    print("3. Fine-tune on your respiratory estimation task")
    

if __name__ == "__main__":
    main()
