#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import pickle as pkl

from utils import *


# In[5]:

EVALUATE = ['models']
SPLIT = ['train', 'test']


# In[3]:


CWD = os.getcwd()
DATASET_DIR = os.path.join(CWD, 'data')
dataset = load_data_from_dir(DATASET_DIR)


# In[4]:


for split in SPLIT:
    for models_type in EVALUATE:
        print(f'evaluating {models_type} on {split}')
        model_dir = os.path.join(CWD, 'best_models', f'final_{models_type}')
        save_to = os.path.join(f'results', split, f'final_{models_type}_results.pkl')
        results = evaluate_models(dataset, model_dir, models_type, split, save_to)

        print(f'saved to {save_to}')


# In[ ]:





# In[ ]:




