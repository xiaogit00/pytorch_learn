############## Transforms ################
'''
All TorchVision datasets have 2 params: transform to modify features, target_transform to modify labels
Accepts callables containing the transformation logic 
'''
#%%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # Make it tensor
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # Make it one hot encoded
)

 # Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, torch.tensor(y), value=1))
 # - for each y, make a row of 0s, and scatter_ will assign value 1 on the index given by y

#%%
(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))([2])