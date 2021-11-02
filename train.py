import torch
import torch.nn as nn
from data import dataset
from model import Model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(x,y, input_size, trainloader, learning_rate=1e-3, epochs = 500, plot=True):

    model = Model(input_size=input_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    trainset = dataset(x,y)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=False)

    losses = []
    accur = []

    for epoch in range(epochs):
        for j,(x_train, y_train) in enumerate(trainloader):

            #output
            output = model(x_train)
            loss = loss_fn(output, y_train.reshape(-1,1))

            #accuracy
            predicted = model(torch.tensor(x, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch%10 == 0:
                losses.append(loss)
                accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(epoch,loss,acc))


            if plot:
                plt.plot(accur)
                plt.title('Accuracy vs Epochs')
                plt.xlabel('epochs')
                plt.ylabel('accuray')
                plt.show()
        