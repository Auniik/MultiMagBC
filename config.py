
import torch
from utils.env import get_base_path, is_runpod

# Get the base path for data depending on the environment
BASE_DATA_PATH = get_base_path()

class Config:
    # Environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12 if is_runpod() else 2

    # Paths
    DATASET_DIR = f"{BASE_DATA_PATH}/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    OUTPUT_DIR = "./output"
    LOGS_DIR = f"{OUTPUT_DIR}/logs"
    MODELS_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"

    # Data
    MAGNIFICATIONS = ['40', '100', '200', '400']
    IMAGE_SIZE = 224

    # K-Fold Cross-Validation
    N_SPLITS = 5
    RANDOM_STATE = 42
    VALIDATION_SPLIT = 0.15
    STRATIFY_SUBTYPE = False

    # Model
    MODEL_NAME = "MultiMagLightweightCNN"
    BASE_CHANNELS = 24
    DROPOUT = 0.3

    # Training - Optimized for 96% accuracy
    BATCH_SIZE = 16  # Reduced for better gradient estimates and stability
    NUM_EPOCHS = 50  # Increased for convergence with early stopping
    LEARNING_RATE = 2e-4  # Slightly higher for faster initial learning
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingWarmRestarts"
    LOSS_FUNCTION = "LabelSmoothingCrossEntropy"
    USE_WEIGHTED_LOSS = True

    @staticmethod
    def get_model_config():
        return {
            "num_classes": 2,
            "base_channels": Config.BASE_CHANNELS,
            "dropout": Config.DROPOUT,
        }

    @staticmethod
    def get_training_config():
        return {
            "batch_size": Config.BATCH_SIZE,
            "num_epochs": Config.NUM_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
            "optimizer": Config.OPTIMIZER,
            "scheduler": Config.SCHEDULER,
            "loss_function": Config.LOSS_FUNCTION,
            "use_weighted_loss": Config.USE_WEIGHTED_LOSS,
        }

# Create an instance of the config
config = Config()
