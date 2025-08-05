#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader


# In[2]:


sys.path.append('src')
from data.dataset import GWDataset, Normalize
from model.handler import ModelHandler
from model.mlp import MLP


# In[3]:


base_data_dir = '/srv/scratch/z5370003/projects/data'
data_dir = os.path.join(base_data_dir, 
                        'groundwater/FEFLOW/coastal/',
                         'variable_density/')
data_path = os.path.join(data_dir, 'all')


# In[4]:


_mean = np.array([1523.10199582, 1729.62046887,   26.72577967,  316.5513978])
_std = np.array([569.13566635, 566.33636362,  14.90088159, 183.2160048 ])
input_transform = Normalize(mean=_mean, std=_std)


# In[5]:


_mean = np.array([3.29223762e-01, 1.78798090e+04])
_std = np.array([2.17284490e-01, 1.51990336e+04])
output_transform = Normalize(mean=_mean, std=_std)


# In[6]:


test_ds = GWDataset(data_path, 'val', 
                     input_transform=input_transform, 
                     output_transform=output_transform,
                     val_ratio=0.3)


test_dl = DataLoader(test_ds, batch_size=10240, 
                      shuffle=False, pin_memory=True)
len(test_dl)


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[8]:


input_size = 4
hidden_sizes = [32, 64, 128, 32, 128, 32]
output_size = 2

mlp = MLP(input_size=input_size,
          hidden_sizes=hidden_sizes,
          output_size=output_size).double()


# In[9]:


model_file = '/srv/scratch/z5370003/projects/04_groundwater/variable_density/logs/saves/mlp_model'
mlp.load_state_dict(torch.load(model_file))
mlp.eval()


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


model_handler = ModelHandler(model=mlp, device=device)


# In[11]:


preds = model_handler.predict(test_dl)


# In[12]:


# print(f"Train loss: {model_handler.evaluate(train_dl)}")
# print(f"Test loss: {model_handler.evaluate(test_dl)}")


# In[ ]:


from tqdm import tqdm

inputs = []
targets = []

for X, y in tqdm(test_dl):
    inputs.append(X.detach().numpy())
    targets.append(y.detach().numpy())


# In[ ]:


write_dir = '/srv/scratch/z5370003/projects/04_groundwater/variable_density/results/MLP/'
np.save(os.path.join(write_dir, 'preds.npy'), preds)
np.save(os.path.join(write_dir, 'inputs.npy'), np.concatenate(inputs))
np.save(os.path.join(write_dir, 'targets.npy'), np.concatenate(targets))


# In[ ]:




