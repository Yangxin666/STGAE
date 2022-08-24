from datetime import datetime
import glob
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
from models.DataAugmentation import *
from models.DataPreprocessing import *
from models.Layers import *
from models.Train import model_train
from models.Test import model_test
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=0.75)
parser.add_argument('--num_nodes', type=int, default=98)
# parser.add_argument('--channels', type=array, default=np.array([[1, 8, 16], [16, 8, 1]]))
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--kernel_size', type=int, default=4)
parser.add_argument('--kernel_size_de', type=int, default=2)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--n_hist', type=int, default=288)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--padding', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--decay_rate', type=float, default=2*e-2)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--train_prop', type=float, default=2/3)
parser.add_argument('--val_prop', type=float, default=1/6)
parser.add_argument('--test_prop', type=float, default=1/6)
parser.add_argument('--normalization', type=str, default='sym')
parser.add_argument('--missing_type', type=str, default='MCAR')
parser.add_argument('--missing_severity', type=float, default=0.2)
parser.add_argument('--bias', type=boolean, default=True)
args = parser.parse_args()
print(f'Training configs: {args}')


### Data Preprocessing
location = pd.read_csv('data/location_35.csv', index_col=0)
W = DataPreprocessing.adjacency_matrix(location, epsilon)


### Data Augmentation
D_O = pd.read_csv('data/norm_power_35.csv')
D_O = D_O.T
D_A = augment(D_O)
corruption_mask = mask(D_0, missing_type, missing_severity)
D_C = corrupt(corruption_mask, D_A)


### Transform Datasets
power_tensor = torch.tensor(D_A.values)
length = D_A.shape[0]
train_x = power_tensor[0:int(train_prop*length),:].to(torch.float32)
validation_x = power_tensor[int(train_prop*length):int((train_prop+val_prop)*length)+1,:].to(torch.float32)
test_x = power_tensor[int((train_prop+val_prop)*length)+1:length,:].to(torch.float32)

power_corrupted_tensor = torch.tensor(D_C.values)
length = D_C.shape[0]
corrupted_train_x = power_corrupted_tensor[0:int(train_prop*length),:].to(torch.float32)
corrupted_validation_x = power_corrupted_tensor[int(train_prop*length):int((train_prop+val_prop)*length)+1,:].to(torch.float32)
corrupted_test_x = power_corrupted_tensor[int((train_prop+val_prop)*length)+1:length,:].to(torch.float32)

device = torch.device("cuda") if torch.cuda.is_available() \
else torch.device("cpu")

x_train, y_train = data_transform(train_x.numpy(), corrupted_train_x.numpy(), n_his, device)
x_val, y_val = data_transform(validation_x.numpy(), corrupted_validation_x.numpy(), n_his, device)
x_test, y_test = data_transform(test_x.numpy(), corrupted_test_x.numpy(), n_his, device)


# create torch data iterables for training
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


# format graph for pyg layer inputs
G = sp.coo_matrix(W)
edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
edge_weight = torch.tensor(G.data).float().to(device)

model_save_path = os.path.join('.data/best_model.pt')
if __name__ == '__main__':
    model_train(train_iter, val_iter, device, args, model_save_path)
    model_test(test_iter, device, args, model_save_path)