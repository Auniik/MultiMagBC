
import argparse
from training.train_k_fold import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Fold training for Multi-Magnification Histopathology model.")
    # You can add arguments here to override config values if needed
    # For example: --epochs 10 --batch_size 64
    args = parser.parse_args()

    # Run the training process
    run_training()
