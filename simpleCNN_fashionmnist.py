#%%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

############## First step: I want to get the data first.  ################
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_images = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
)
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

############## Second step: I want to feem them into dataloaders ################

batch_size = 64
train_dataloader= DataLoader(training_data, batch_size=batch_size)
test_dataloader= DataLoader(test_data, batch_size=batch_size)

############## Third step: I want to define a neural network.  ################
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2, stride=1, padding=0) #(64, 10, 27, 27)
        self.bn = nn.BatchNorm2d(10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #(64, 10, 13, 13)
        self.flatten = nn.Flatten() #(64, 13*13*10)
        self.fc1 = nn.Linear(13*13*10, 10) # I expect input of shape (64, 1, 28 28)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        logits = self.fc1(x)
        return logits

#%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
predictor = SimpleCNN().to(device)
#%%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
#%%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 8
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, predictor, loss_fn, optimizer)
    test(test_dataloader, predictor, loss_fn)
print("Done!")
# %%
