#%%
# Run this cell, and the cells below to predict!
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_images = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
)
class ShallowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        Z1 = self.fc1(x)
        A1 = torch.relu(Z1)
        logits = self.fc2(A1)
        return logits

predictor = ShallowNetwork().to(device)

predictor.load_state_dict(torch.load("predictors/fashionMNIST_shallow_nn_predictor.pth", weights_only=True))

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
#%%
n = 1323
X, Y = test_data[n][0].to(device), torch.tensor([test_data[n][1]]).to(device)
test_images[n][0]
#%%
classes[test_images[n][1]]
#%%
print("Prediction is:", classes[predictor(X).argmax(1)])
if predictor(X).argmax(1)==Y:
    print("Correct!")
else:
    print("Wrong")

# %%
