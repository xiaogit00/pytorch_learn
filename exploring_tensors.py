#%%
import torch
import numpy as np

#%% CREATING TENSORS
data = [[1, 2], [3, 4]]
torch.tensor(data) #Creating tensor
# %%
np_array = np.array(data)
x_np = torch.from_numpy(np_array) #Creating tensor from numpy array
x_np
# %%
x_ones = torch.ones_like(x_np) # Creating similar tensors
x_ones
#%%
x_rand = torch.rand_like(x_np, dtype=torch.float) # Overrides datatype of x_data, [0,1]
x_rand

# %%
shape = (2, 3,) # a tuple of dimensions
rand_tensor = torch.rand(shape)
rand_tensor
# %% TENSOR ATTRIBUTES
tns = torch.rand(3, 4)
tns.shape
# %%
tns.dtype
# %%
tns.device
# %%
tns.to("mps")
# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# %%
tns.to(device)
tns.device
# %%
tns2 = torch.rand(4,4)
tns2
# %% TENSOR SLICING
tns2[:, 0] # First column
# %%
tns2[0] # First row 
# %%
tns2[..., -1] # Last column
'''
... means “expand to however many : slices are needed to cover all the remaining dimensions.”
x = torch.arange(2*3*4).reshape(2, 3, 4)
x[..., -1] → shape [2, 3]
It selects the last element in the last dimension (size 4 → index -1).
Equivalent to x[:, :, -1].
'''
# %%
x = torch.arange(2*3*4).reshape(2, 3, 4)
x[:,:,-1].shape

# %%
x[:, -1]
# %%
x[..., -1]
# %% JOINING TENSORS
tns3 = torch.cat([tns, tns], dim=1) # Join horizontally
tns3.shape

# %%
tns4 = torch.cat([tns, tns, tns], dim=0) # Join vertically
tns4.shape

#%%
y = tns @ tns.T
y
# %%
y2 = tns.matmul(tns.T)
y2
# %%
tns.sum()
# %% IN PLACE OPERATIONS
tns

# %%
tns.add_(5)
# %%
tns
# %%
tns + 5
# %%
tns
# %% Memory sharing between Tensors & Numpy
t = torch.ones(5)
n = t.numpy()
t, n

# %%
t.add_(5)
# %%
n