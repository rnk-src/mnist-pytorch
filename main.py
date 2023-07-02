import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
from datetime import datetime

seed = 100
torch.manual_seed(seed)

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]))
mnist_train = data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=True, num_workers=0)

mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]))
mnist_test = data.DataLoader(dataset=mnist_test, batch_size=32, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3
epochs = 10


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Starting with 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)  # 4x28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)  # 8x28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 8x14x14
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)  # 16x14x14
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 32x7x7
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_run_folder():
    now = datetime.now()
    now = now.strftime("%b-%d-%Y %H:%M:%S")
    os.mkdir("runs/run " + now)


make_run_folder()


def save_checkpoint(state):
    pass


model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loss_array = []
test_loss_array = []
epoch_array = []
total_loss = 0

for epoch in range(epochs):

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for inputs, labels in mnist_train:
        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
        print("Training - Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print("Total Train loss on epoch " + str(epoch) + ": " + str(total_loss))
    train_loss_array.append(total_loss)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in mnist_test:
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            print("Testing - Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
            total_loss += loss.item()
    print("Total Test loss on epoch " + str(epoch) + ": " + str(total_loss))
    test_loss_array.append(total_loss)
    total_loss = 0
    epoch_array.append(epoch)
