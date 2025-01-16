# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:55:01 2025

@author: lenovo
"""

import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataloader
from torch.utils.data import Dataset, DataLoader
#from torch_geometric.loader import DataLoader
#from torch_geometric.nn import GATConv,GCNConv
import matplotlib.pyplot as plt

from sklearn import metrics
import numpy as np


def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch[0].cuda(),batch[0].cuda())
            #rint(batch.y.size())
            y = batch[1].cuda()
            #print(out.shape)
            #print(y.shape)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
        
def pred(model,test_loader):
    out_list=[]
    true_list=[]
    for batch in test_loader:
        out=model(batch[0].cuda(),batch[0].cuda())
        #print(out.shape)
        true=batch[1].cuda()
        out_list.append(out.cpu().detach().numpy())
        true_list.append(true.cpu().detach().numpy())
    
    return np.vstack(out_list),np.vstack(true_list)

def inverse_transform(x):
    
    result=[]
    
    for i in range(x.shape[0]):
        #print(x[0].shape)
        #y=x[i].reshape(x[i].shape[0],-1,1)
        y_inverse=x[i][0]
        result.append(y_inverse)
    
    return np.vstack(result)