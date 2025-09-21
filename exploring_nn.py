############## nn ################
'''
The torch.nn namespace provides all the building blocks you need to build your network. 
Every module subclasses the nn.Module -> a neural network is a module itself and consists of other modules (layers)
'''
#%%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
############## On the forward method ################
'''
To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly!
'''

############## nn.Sequential ################
'''
An ordered container of modules. The data is passed through all the modules in the same order defined. -> almost like threading. 
'''

model = NeuralNetwork()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}.grad -> {param.grad.shape}")
        print(param.grad)  
    else:
        print(f"{name}.grad is None")