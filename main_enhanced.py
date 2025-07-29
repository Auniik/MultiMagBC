#!/usr/bin/env python3
"""
Enhanced Main Training Script for 96% Accuracy Target
====================================================

This script runs the enhanced training pipeline with:
- Advanced data utilization (all available images per patient)
- Sophisticated augmentation strategies
- Anti-overfitting techniques (label smoothing, early stopping, dropout scheduling)
- Test-time augmentation for maximum performance
- Comprehensive monitoring and evaluation

Usage:
    python main_enhanced.py [--quick-test] [--no-tta] [--single-fold]

Arguments:
    --quick-test: Run with reduced settings for quick testing
    --no-tta: Disable test-time augmentation
    --single-fold: Run only the first fold for testing
"""

import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.enhanced_train_k_fold import run_enhanced_training
from config import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Magnification Histopathology Training for 96% Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_enhanced.py                    # Full training with all enhancements
    python main_enhanced.py --quick-test       # Quick test run
    python main_enhanced.py --single-fold      # Test single fold only
    python main_enhanced.py --no-tta           # Disable test-time augmentation
        """
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run with reduced epochs and batch size for quick testing'
    )
    
    parser.add_argument(
        '--no-tta', 
        action='store_true',
        help='Disable test-time augmentation'
    )
    
    parser.add_argument(
        '--single-fold', 
        action='store_true',
        help='Run only the first fold for testing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def modify_config_for_testing(args):
    """Modify configuration based on arguments"""
    if args.quick_test:
        print("🚀 Quick test mode enabled")
        # Don't modify config directly, create a new config dict
        pass
    
    if args.single_fold:
        print("📋 Single fold mode enabled")
        # Don't modify config directly, this will be handled in run_enhanced_training
        pass
    
    if args.no_tta:
        print("⚡ TTA disabled")
        # This will be handled in the trainer


def print_configuration():
    """Print current configuration"""
    print("\n" + "="*60)
    print("🎯 ENHANCED TRAINING CONFIGURATION")
    print("="*60)
    print(f"📊 Model: {config.MODEL_NAME}")
    print(f"🔧 Base Channels: {config.BASE_CHANNELS}")
    print(f"💧 Dropout: {config.DROPOUT}")
    print(f"📚 Batch Size: {config.BATCH_SIZE}")
    print(f"🔄 Epochs: {config.NUM_EPOCHS}")
    print(f"📈 Learning Rate: {config.LEARNING_RATE}")
    print(f"🎲 K-Folds: {config.N_SPLITS}")
    print(f"🖼️  Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"🔍 Magnifications: {config.MAGNIFICATIONS}")
    print(f"💾 Device: {config.DEVICE}")
    print(f"👥 Workers: {config.NUM_WORKERS}")
    print("="*60)


def check_environment():
    """Check if environment is properly set up"""
    print("\n🔍 Environment Check:")
    
    # Check data directory
    if not os.path.exists(config.DATASET_DIR):
        print(f"❌ Dataset directory not found: {config.DATASET_DIR}")
        print("   Please ensure the BreakHis dataset is properly downloaded and extracted.")
        return False
    
    # Check for some sample data
    sample_dirs = [
        os.path.join(config.DATASET_DIR, "benign"),
        os.path.join(config.DATASET_DIR, "malignant")
    ]
    
    for sample_dir in sample_dirs:
        if not os.path.exists(sample_dir):
            print(f"❌ Missing expected directory: {sample_dir}")
            return False
    
    print("✅ Dataset directory structure looks good")
    
    # Check output directories will be created
    print(f"✅ Output directory: {config.OUTPUT_DIR}")
    
    # Check device
    import torch
    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        config.DEVICE = "cpu"
    
    print(f"✅ Using device: {config.DEVICE}")
    
    if config.DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True


def main():
    """Main function"""
    print("🔬 Enhanced Multi-Magnification Histopathology Training")
    print("🎯 Target: 96% Accuracy with Anti-Overfitting")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Modify configuration
    modify_config_for_testing(args)
    
    # Print configuration
    print_configuration()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Show key enhancements
    print("\n🚀 KEY ENHANCEMENTS ENABLED:")
    print("   ✅ Advanced data utilization (all patient images)")
    print("   ✅ Sophisticated augmentation pipeline")
    print("   ✅ Label smoothing for better generalization")
    print("   ✅ Early stopping to prevent overfitting")
    print("   ✅ Learning rate scheduling with warm restarts")
    print("   ✅ Gradient clipping for stability")
    print("   ✅ Class-weighted loss for imbalance handling")
    print("   ✅ Comprehensive metrics tracking")
    
    if not args.no_tta:
        print("   ✅ Test-time augmentation for maximum accuracy")
    else:
        print("   ⚠️  Test-time augmentation disabled")
    
    # Confirm before starting
    if not args.quick_test:
        print(f"\n⚡ This will run {config.N_SPLITS} folds with {config.NUM_EPOCHS} epochs each.")
        print("   This may take several hours depending on your hardware.")
        
        response = input("\n🚀 Start enhanced training? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("⏹️  Training cancelled.")
            sys.exit(0)
    
    try:
        print("\n" + "="*60)
        print("🎯 STARTING ENHANCED TRAINING")
        print("="*60)
        
        # Disable TTA in trainer if requested
        if args.no_tta:
            # This would need to be passed to the trainer
            pass
        
        # Run enhanced training
        run_enhanced_training(args)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show results location
        print(f"📊 Results saved to: {config.RESULTS_DIR}")
        print(f"📈 Logs saved to: {config.LOGS_DIR}")
        print(f"💾 Models saved to: {config.MODELS_DIR}")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()