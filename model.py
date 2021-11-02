import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size):
        super(self, Model).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.lstm = nn.LSTM(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.lstm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x