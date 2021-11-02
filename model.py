import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.transpose(input=x, dim0=0, dim1=1)
        x = torch.relu(x)

        print("Shape after fc1: ", x.shape)

        x = torch.transpose(input=x, dim0=0, dim1=1)

        x = self.fc2(x)
        x = torch.transpose(input=x, dim0=0, dim1=1)
        x = torch.relu(x)

        print("Shape after fc2: ", x.shape)

        x = self.fc3(x)
        x = torch.transpose(input=x, dim0=0, dim1=1)
        x = torch.relu(x)

        print("Shape after fc3: ", x.shape)

        x = self.fc4(x)
        x = torch.transpose(input=x, dim0=0, dim1=1)
        x = torch.relu(x)

        print("Shape after fc4: ", x.shape)

        x = torch.transpose(input=x, dim0=0, dim1=1)
        x = self.fc5(x)
        x = torch.transpose(input=x, dim0=0, dim1=1)

        print("Shape after fc5: ", x.shape)

        x = torch.sigmoid(x)

        x = torch.transpose(input=x, dim0=0, dim1=1)

        return x