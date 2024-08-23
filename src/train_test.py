import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs import *
from .utils import early_stopping

################################################################################

def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer, 
          device: str,
          epochs: int,
          scheduler: lr_scheduler=None,) -> list:
    
    model.train()
    train_loss_list = []
    val_loss_list = []
    best_loss = float("inf")
    counter = EARLY_STOPPING_COUNTER

    for epoch in epochs:
        total_train_loss = 0
        total_val_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch [{epoch}/{epochs}]") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if scheduler: scheduler.step()

            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                val_output = model(data)
                val_loss = criterion(val_output, target)
                total_val_loss += val_loss.item()

                if USE_EARLY_STOPPING:
                    flag, counter, best_loss = early_stopping(val_loss=val_loss.item(), 
                                                              patience=5, 
                                                              best_loss=best_loss, 
                                                              counter=counter)
                    if flag: 
                        torch.save(model.state_dict(), f"{EXPERIMENT_NAME}_best.pth")
                        break
                    
                    counter = counter
                    best_loss = best_loss

            train_loss_list.append(total_train_loss)
            val_loss_list.append(total_val_loss)
            pbar.set_postfix(loss=total_train_loss, val_loss=total_val_loss)

    return train_loss_list, val_loss_list

################################################################################

@torch.no_grad()
def test(model: nn.Module,
         dataloader: DataLoader,
         criterion: nn.Module,
         device: str) -> float:
    
    model.eval()
    correct = 0
    total = 0
    loss = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss += criterion(output, target).item()

        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        accuracy = correct/total
        total_loss = loss/total

        print(f"Accuracy: {accuracy}, Loss: {total_loss}")
    
    return accuracy, total_loss