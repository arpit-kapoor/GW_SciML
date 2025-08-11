#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os, sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchinfo import summary


# In[2]:


sys.path.append('src')
from src.data.points_dataset import GWDataset
from src.data.transform import Normalize
from src.models.handler import ModelHandler
from src.models.mlp import MLP


# In[3]:


base_data_dir = '/srv/scratch/z5370003/projects/data'
data_dir = os.path.join(base_data_dir, 
                        'groundwater/FEFLOW/coastal/',
                         'variable_density/')
data_path = os.path.join(data_dir, 'filter')


# In[4]:


_mean = np.array([1523.10199582, 1729.62046887,   26.72577967,  316.5513978])
_std = np.array([569.13566635, 566.33636362,  14.90088159, 183.2160048 ])
input_transform = Normalize(mean=_mean, std=_std)


# In[5]:


_mean = np.array([3.29223762e-01, 1.78798090e+04])
_std = np.array([2.17284490e-01, 1.51990336e+04])
output_transform = Normalize(mean=_mean, std=_std)


# In[6]:


print(f"Loading data from path {data_path}...")

train_ds = GWDataset(data_path, 'train', 
                     input_transform=input_transform, 
                     output_transform=output_transform)

print(f"Loaded Train data with {len(train_ds)} samples from {data_path}.")


# In[8]:


train_dl = DataLoader(train_ds, batch_size=4096, 
                      shuffle=True, pin_memory=True)



# In[14]:


test_ds = GWDataset(data_path, 'val', 
                     input_transform=input_transform, 
                     output_transform=output_transform)

print(f"Loaded Test data with {len(test_ds)} samples from {data_path}.")


test_dl = DataLoader(test_ds, batch_size=4096, 
                      shuffle=False, pin_memory=True)



# In[9]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[10]:


input_size = 4
hidden_sizes = [32, 64, 128, 32, 128, 32]
output_size = 2

mlp = MLP(input_size=input_size,
          hidden_sizes=hidden_sizes,
          output_size=output_size)

opt = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)

# print model summary
print(summary(mlp, (1, input_size)))

mlp = mlp.double()



# In[11]:
model_handler = ModelHandler(model=mlp, device=device, optimizer=opt)


# In[12]:

print("Training model...")
model_handler.train(train_dl, num_epochs=50)
model_handler.save_model('logs/saves/mlp_model')


# In[ ]:
print(f"Train loss: {model_handler.evaluate(train_dl)}")
print(f"Test loss: {model_handler.evaluate(test_dl)}")

