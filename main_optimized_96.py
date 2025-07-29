#!/usr/bin/env python3
"""
Optimized Training Script for 96% Accuracy Target
================================================

This script runs the fully optimized training pipeline with all enhancements:
- Efficient TTA implementation
- Enhanced model capacity (32 base channels)
- Progressive dropout scheduling
- Focal loss for hard examples
- Statistical significance testing
- Ensemble predictions
- All security and warning fixes

Usage:
    python main_optimized_96.py [--focal-loss] [--quick-test]
"""

import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.enhanced_train_k_fold import run_enhanced_training, AdvancedTrainer
from config import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Optimized Multi-Magnification Training for 96% Accuracy Target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_optimized_96.py                 # Full optimized training
    python main_optimized_96.py --focal-loss    # Use focal loss for hard examples
    python main_optimized_96.py --quick-test    # Quick test run
        """
    )
    
    parser.add_argument(
        '--focal-loss', 
        action='store_true',
        help='Enable focal loss for hard examples and class imbalance'
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run with reduced settings for quick testing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_optimizations():
    """Print all active optimizations"""
    print("\n🚀 ACTIVE OPTIMIZATIONS FOR 96% ACCURACY:")
    print("="*60)
    print("✅ Efficient TTA (no dataset recreation)")
    print("✅ Enhanced model capacity (32 base channels)")
    print("✅ Progressive dropout scheduling (0.2 → 0.6)")
    print("✅ Increased validation split (20%)")
    print("✅ Stratified subtype splitting")
    print("✅ Security fixes (weights_only=True)")
    print("✅ Zero division handling for metrics")
    print("✅ Early stopping (patience=12)")
    print("✅ Gradient clipping (max_norm=1.0)")
    print("✅ Statistical significance testing")
    print("✅ Ensemble predictions across folds")
    print("✅ Advanced loss functions available")
    print("="*60)


def main():
    """Main function with all optimizations"""
    print("🎯 OPTIMIZED TRAINING FOR 96% ACCURACY TARGET")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Print active optimizations
    print_optimizations()
    
    # Special configurations for 96% target
    if args.focal_loss:
        print("\n🔥 FOCAL LOSS ENABLED")
        print("   - Better handling of hard examples")
        print("   - Improved class imbalance handling")
        print("   - Alpha=1.0, Gamma=2.0")
    
    if args.quick_test:
        print("\n🚀 QUICK TEST MODE")
        print("   - 5 epochs instead of 50")
        print("   - Batch size 8 instead of 16")
    
    # Enhanced configuration summary
    print(f"\n📊 ENHANCED CONFIGURATION:")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Base Channels: {config.BASE_CHANNELS} (enhanced from 24)")
    print(f"   Dropout: {config.DROPOUT} (enhanced from 0.3)")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   Validation Split: {config.VALIDATION_SPLIT} (enhanced from 0.15)")
    print(f"   Stratify Subtype: {config.STRATIFY_SUBTYPE} (enabled)")
    print(f"   K-Folds: {config.N_SPLITS}")
    
    # Confirm training
    if not args.quick_test:
        print(f"\n⚡ This will run {config.N_SPLITS} folds with enhanced settings.")
        print("   Expected improvements:")
        print("   - Better statistical significance")
        print("   - Reduced overfitting")
        print("   - Higher peak accuracy")
        print("   - More stable results")
        
        response = input("\n🚀 Start optimized training for 96% target? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("⏹️  Training cancelled.")
            sys.exit(0)
    
    try:
        print("\n" + "="*60)
        print("🎯 STARTING OPTIMIZED TRAINING")
        print("="*60)
        
        # Configure advanced trainer if focal loss requested
        class OptimizedTrainer(AdvancedTrainer):
            def __init__(self, config, use_focal_loss=False):
                super().__init__(config)
                if use_focal_loss:
                    self.use_focal_loss = True
                    self.use_label_smoothing = False  # Use focal instead
                    print("🔥 Focal loss enabled for hard examples")
        
        # Monkey patch the trainer creation
        original_trainer_init = AdvancedTrainer.__init__
        def enhanced_trainer_init(self, config):
            original_trainer_init(self, config)
            if args.focal_loss:
                self.use_focal_loss = True
                self.use_label_smoothing = False
                print("🔥 Focal loss enabled for hard examples")
        
        AdvancedTrainer.__init__ = enhanced_trainer_init
        
        # Run enhanced training
        run_enhanced_training(args)
        
        print("\n" + "="*60)
        print("🎉 OPTIMIZED TRAINING COMPLETED!")
        print("="*60)
        
        # Show results location
        print(f"📊 Results saved to: {config.RESULTS_DIR}")
        print(f"📈 Logs saved to: {config.LOGS_DIR}")
        print(f"💾 Models saved to: {config.MODELS_DIR}")
        
        print("\n💡 WHAT TO CHECK:")
        print("   1. Look for TTA results ≥96% in any fold")
        print("   2. Check ensemble predictions across folds")
        print("   3. Review confidence intervals for statistical significance")
        print("   4. Monitor for reduced overfitting vs previous runs")
        
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