#%%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


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

#%%
batch_size = 64
train_dataloader= DataLoader(training_data, batch_size=batch_size)
test_dataloader= DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class ShallowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("Performing Forward pass...")
        # print("Receiving input of shape:", x.shape)
        x = self.flatten(x)
        # print("Flattening:", x.shape)
        Z1 = self.fc1(x)
        # print("Z1:", Z1.shape)
        A1 = torch.relu(Z1)
        # print("A1:", A1.shape)
        logits = self.fc2(A1)
        # print("logits:", logits)
        # prob_dist = self.softmax(logits)
        return logits

predictor = ShallowNetwork().to(device)
#%%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)
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

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, predictor, loss_fn, optimizer)
    test(test_dataloader, predictor, loss_fn)
print("Done!")

torch.save(predictor.state_dict(), "predictors/fashionMNIST_shallow_nn_predictor.pth")