import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# import wandb
import optuna
import matplotlib.pyplot as plt

from model import NeuralNetwork
from dataset import CustomDataset
from configs import *
from train_test import train, test
from utils import plot_figure, set_seed, set_logger

################################################################################

def main():
    if SET_SEED: set_seed(SEED)
    if SET_MANUAL_LOGGER: set_logger(LOG_PATH)

    Net = NeuralNetwork(num_classes=NUM_CLASSES, 
                        input_channels=3, 
                        input_size=32, 
                        hidden_size=128, 
                        dropout=0.5)
    Net.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(Net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if USE_SCHEDULER:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    else:
        scheduler = None
    
    train_val_dataset = CustomDataset()
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_dataset = CustomDataset()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if LOAD_MODEL:
        Net.load_state_dict(torch.load(CHECKPOINT_FILE))
    
    train_loss, val_loss = train(Net, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS, scheduler)
    accuracy, test_loss = test(Net, test_loader, criterion, DEVICE)

    if SAVE_LOSS_FIGURE:
        plot_figure(train_loss, val_loss, title="Training Loss", save_path=LOSS_FIGURE_SAVE_PATH)
    if SAVE_MODEL:
        torch.save(Net.state_dict(), CHECKPOINT_FILE)

    print(f"Accuracy: {accuracy}")
    print(f"Loss: {test_loss}")

def objective(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 512)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    Net = NeuralNetwork(num_classes=NUM_CLASSES, 
                        input_channels=3, 
                        input_size=32, 
                        hidden_size=hidden_size, 
                        dropout=dropout)
    Net.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(Net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if USE_SCHEDULER:
        step_size = trial.suggest_int("step_size", 1, 10)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    train_val_dataset = CustomDataset()
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_dataset = CustomDataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    for epoch in range(EPOCHS):
        train_loss, val_loss = train(Net, train_loader, val_loader, criterion, optimizer, DEVICE, 1, scheduler)
        accuracy, test_loss = test(Net, test_loader, criterion, DEVICE)

        trial.report(test_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return test_loss

################################################################################

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best Params: ", study.best_params)
    #main()