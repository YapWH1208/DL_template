import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_size=32, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
        )
    
    def forward(self, x):
        return x