from models.layers import *
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch_geometric.nn import ChebConv

def model_train(train_iter, val_iter, device, args, model_saved_path):
    model = STConvAE(device, num_nodes, channels, num_layers, kernel_size, K, n_his, kernel_size_de, stride, padding, normalization = 'sym', bias = True).to(device)
    # define loss function
    loss = nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.02)

    min_valid_loss = np.inf

    for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch', position=0):
        train_loss, n = 0.0, 0
        model.train()

        for x, y in tqdm(train_iter, desc='Batch', position=0):
            # get model predictions and compute loss
            y_pred = model(x.to(device), edge_index, edge_weight)
            loss = torch.mean((y_pred - y) ** 2)
            # backpropogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()
        for x, y in tqdm(val_iter, desc='Batch', position=0):
            # get model predictions and compute loss
            y_pred = model(x.to(device), edge_index, edge_weight)
            loss = torch.mean((y_pred - y) ** 2)
            valid_loss += loss.item()

        print(f'Epoch {epoch} \t\t Training Loss: {train_loss / 120} \t\t Validation Loss: {valid_loss / 30}')
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)

    print('Training model finished!')