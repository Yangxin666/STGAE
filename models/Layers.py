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


class TemporalConv(nn.Module):

    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int, padding: int):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0,padding))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0,padding))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0,padding))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H



class TemporalDeConv1(nn.Module):

    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int, padding: int):
        super(TemporalDeConv1, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0,padding))
        self.conv_2 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size),(1, stride), (0,padding))
        self.conv_3 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size),(1, stride), (0,padding))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class TemporalDeConv2(nn.Module):

    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int):
        super(TemporalDeConv2, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size),(1, stride))
        self.conv_2 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size),(1, stride))
        self.conv_3 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size),(1, stride))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConvEncoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConvEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size, stride = stride, padding = padding,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride = stride, padding = padding,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,) -> torch.FloatTensor:

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        #print(X.shape)
        T_0 = self._temporal_conv1(X)
        #print(T_0.shape)
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

        T = F.relu(T)
        #print(T.shape)
        T = self._temporal_conv2(T)
        #print(T.shape)
        # T = T.permute(0, 2, 1, 3)
        # #print(T.shape)
        # T = self._batch_norm(T)
        # T = T.permute(0, 2, 1, 3)
        #print(T.shape)

        return T

class STConvDecoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        kernel_size_de: int,
        stride: int,
        padding: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConvDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalDeConv1(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size, stride = stride, padding = padding,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalDeConv2(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_de, stride = stride,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,) -> torch.FloatTensor:

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        # T = T.permute(0, 2, 1, 3)
        # T = self._batch_norm(T)
        # T = T.permute(0, 2, 1, 3)
        return T


# a specified number of STConv blocks, followed by an output layer
class STConvAE(torch.nn.Module):
    def __init__(self, device, num_nodes, channel_size_list, num_layers,
                 kernel_size, K, window_size, kernel_size_de, stride, padding, \
                 normalization='sym', bias=True):
        # num_nodes = number of nodes in the input graph
        # channel_size_list =  2d array representing feature dimensions throughout the model
        # num_layers = number of STConv blocks
        # kernel_size = length of the temporal kernel
        # K = size of the chebyshev filter for the spatial convolution
        # window_size = number of historical time steps to consider

        super(STConvAE, self).__init__()
        self.layers = nn.ModuleList([])
        # add STConv blocks
        for l in range(num_layers):
            input_size, hidden_size, output_size = channel_size_list[l][0], channel_size_list[l][1], \
                                                   channel_size_list[l][2]
            if l == 0:
                self.layers.append(
                    STConvEncoder(num_nodes, input_size, hidden_size, output_size, kernel_size, stride, padding, K,
                                  normalization, bias))
            if l == 1:
                self.layers.append(
                    STConvDecoder(num_nodes, input_size, hidden_size, output_size, kernel_size, kernel_size_de, stride,
                                  padding, K, normalization, bias))

        # # add output layer
        # self.layers.append(OutputLayer(channel_size_list[-1][-1], \
        #                                window_size - 2 * num_layers * (kernel_size - 1), \
        #                                num_nodes))
        # CUDA if available
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        # print(x.shape)
        # out_layer = self.layers[-1]
        # x = x.permute(0, 3, 1, 2)
        # x = out_layer(x)
        # print(x.shape)
        return x



