def early_stopping(val_loss:float, 
                   patience:int, 
                   best_loss:float, 
                   counter:int) -> tuple[bool, int, float]:
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print(f"Early Stopping. Val loss: {val_loss:.3f} Best Val loss: {best_loss:.3f}")
        return True, counter, best_loss
    return False, counter, best_loss

################################################################################

def plot_figure(train_loss:list[float], 
                val_loss:list[float],
                title:str="Training Loss",
                save_path:str="train_loss.png") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 8))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)

################################################################################

def set_seed(seed:int=42) -> None:
    import torch
    import numpy as np
    import random
    import os

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

################################################################################

def set_logger(log_path):
    import os
    import logging

	# remove the log with same name
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    # Initialize log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)