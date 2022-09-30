#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：dcn_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/14 8:54 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch
import torch.nn as nn
from dataset import get_sparse_fea_num
import torch.nn.functional as F

import config


class Cross(nn.Module):
    def __init__(self, num_layers, cross_dim):
        super(Cross, self).__init__()
        self.num_layers = num_layers
        # 在这依然像之前pnn中一样采用初始化w参数矩阵的方式，而是直接定义了线性层，在我目前来看这样更为清晰明了
        self.linear = nn.ModuleList([nn.Linear(cross_dim, 1) for _ in range(num_layers)])
        self.bias = nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(cross_dim,1))) for _ in range(num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2) # [batch_size,feature_nums,1]
        x_l = x_0
        for i in range(self.num_layers):
            # 这一步其实就是每个特征与其他特征（包括自己）做交叉的过程
            tmp = x_0.matmul(x_l.transpose(2, 1)).to(torch.float32) # tmp:[batch_size,feature_num,feature_num]
            tmp = self.linear[i](tmp)
            # 这一步类似于跳跃连接
            x_l = tmp + self.bias[i] + x_l

        return x_l.squeeze(2)


class DNN(nn.Module):
    def __init__(self, hidden_units, drop_out):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = x.float()
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        return self.drop_out(x)


class DeepCross(nn.Module):
    def __init__(self, hidden_units=config.hidden_units, drop_out=config.dropout,
                 embedding_dim=config.embedding_dim):
        super(DeepCross, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + str(key): nn.Embedding(val, embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        dnn_cross_dim = len(self.sparse) * embedding_dim + len(self.dense)
        hidden_units_copy = hidden_units.copy()
        hidden_units_copy.insert(0,dnn_cross_dim)
        self.dnn = DNN(hidden_units=hidden_units_copy, drop_out=drop_out)
        self.cross = Cross(2, dnn_cross_dim)
        self.final_linear = nn.Linear(dnn_cross_dim + hidden_units_copy[-1], 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedded = [self.embedding_layers['embedding_' + col_name](sparse_inputs[:, col_val]) for
                           col_name, col_val in
                           zip(self.sparse_fea_num.keys(), range(sparse_inputs.shape[1]))]
        sparse_embedded = torch.cat(sparse_embedded, dim=-1)
        cross_deep_inputs = torch.cat((dense_inputs, sparse_embedded), dim=-1)

        # cross
        cross_outputs = self.cross(cross_deep_inputs)
        deep_outputs = self.dnn(cross_deep_inputs)

        cd_outputs = self.final_linear(torch.cat((cross_outputs, deep_outputs), dim=-1).float())

        return torch.sigmoid(cd_outputs).squeeze(-1)
