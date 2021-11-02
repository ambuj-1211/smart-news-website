import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, input_size):
        super(self, Model).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, 32)

    def forward(self, x):
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.lstm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x