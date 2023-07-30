import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
from datetime import datetime
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

model_load = False
seed = 100
torch.manual_seed(seed)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
fig.tight_layout(pad=4.0)

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]))
mnist_train, mnist_validation = train_test_split(mnist_train, test_size=0.25)

mnist_train = data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=True, num_workers=0)
mnist_validation = data.DataLoader(dataset=mnist_validation, batch_size=64, shuffle=True, num_workers=0)

mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]))
mnist_test = data.DataLoader(dataset=mnist_test, batch_size=64, shuffle=True, num_workers=0)

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


def get_current_time():
    now = datetime.now()
    now = now.strftime("%b-%d-%Y %H:%M:%S")
    return now


filename = "run " + get_current_time()
run_directory = "runs/" + filename
os.mkdir(run_directory)


def save_checkpoint(state, directory=run_directory):
    torch.save(state, directory + "/checkpoint " + get_current_time() + ".pt")
    print("saved checkpoint")


def get_accuracy(out, lbl):
    n = out.size(0)
    out = torch.softmax(out, dim=1)
    max_scores, max_idx_class = out.max(
        dim=1)
    return (max_idx_class == lbl).sum().item() / n


model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)


def load_model(path):
    load_dict = torch.load(path)
    model.load_state_dict(load_dict['model_state_dict'])
    optimizer.load_state_dict(load_dict['optimizer_state_dict'])


if model_load:
    load_model("runs/run Jul-04-2023 23:27:12/checkpoint Jul-05-2023 01:19:35.pt")
    model.train()

train_loss_array = []
train_accuracy_array = []
validation_loss_array = []
validation_accuracy_array = []
epoch_array = []
total_loss = 0

for epoch in range(epochs):
    model.train()
    temp_accuracy_array = []
    for inputs, labels in mnist_train:
        optimizer.zero_grad()
        output = model(inputs)
        temp_accuracy_array.append(get_accuracy(output, labels))
        loss = criterion(output, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_accuracy_array.append(sum(temp_accuracy_array)/len(temp_accuracy_array))
    temp_accuracy_array = []

    print("Total Train loss on epoch " + str(epoch) + ": " + str(total_loss))
    train_loss_array.append(total_loss)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in mnist_validation:
            output = model(inputs)
            temp_accuracy_array.append(get_accuracy(output, labels))
            loss = criterion(output, labels)
            total_loss += loss.item()
    validation_accuracy_array.append(sum(temp_accuracy_array)/len(temp_accuracy_array))

    print("Total validation loss on epoch " + str(epoch) + ": " + str(total_loss))
    validation_loss_array.append(total_loss)
    total_loss = 0
    epoch_array.append(epoch)

    scheduler.step()

    checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}
    save_checkpoint(checkpoint)

load_model(str(input("Directory for best epoch: ")))
model.eval()

test_accuracy_array = []
with torch.no_grad():
    for inputs, labels in mnist_test:
        outputs = model(inputs)
        num_correct = 0
        test_accuracy_array.append(get_accuracy(outputs, labels))

print("Final accuracy: " + str(sum(test_accuracy_array) / len(test_accuracy_array)))

ax1.plot(epoch_array, train_loss_array, label='Training Loss', linestyle='dashed', color='red')
ax1.plot(epoch_array, validation_loss_array, label='Validation Loss', linestyle='dotted', color='blue')
ax1.legend()
ax1.set_title('Losses by Epoch')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Losses')

ax2.plot(epoch_array, train_accuracy_array, label='Training Accuracy', linestyle='dashed', color='red')
ax2.plot(epoch_array, validation_accuracy_array, label='Validation Accuracy', linestyle='dotted', color='blue')
ax2.legend()
ax2.set_title('Accuracy by Epoch')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.ylim(0, 1)

plt.show()
