import torch
from time import time

# Hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10

# Loaders
NUM_WORKERS = 4
PIN_MEMORY = True

# Load and save model
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SCHEDULER = False
EXPERIMENT_NAME = "Experiment" + f"{time()}"

# If you want to save loss figure, set SAVE_LOSS_FIGURE to True
SAVE_LOSS_FIGURE = True
LOSS_FIGURE_SAVE_PATH = "train_loss.png"

# If you want to use early stopping, set USE_EARLY_STOPPING to True
USE_EARLY_STOPPING = False
EARLY_STOPPING_COUNTER = 5

# If you want to use early stopping, set USE_EARLY_STOPPING to True
SET_SEED = True
SEED = 42

# If you want to use manual logger, set SET_MANUAL_LOGGER to True
# Need to change print() to logger.info()
SET_MANUAL_LOGGER = False
LOG_PATH = "train.log"