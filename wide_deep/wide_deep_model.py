#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：wide_deep_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/14 5:51 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch
import torch.nn as nn
from dataset import get_sparse_fea_num
import config
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        x = x.float()
        return self.linear(x)


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


class WideDeep(nn.Module):

    def __init__(self, hidden_units=config.hidden_units, drop_out=config.dropout):
        super(WideDeep, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        # embedding
        self.embed_layers = nn.ModuleDict(
            {'embed_' + str(key): nn.Embedding(val, config.embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        hidden_units_copy = hidden_units.copy()
        hidden_units_copy.insert(0, len(self.dense) + (len(self.sparse) * config.embedding_dim))
        self.dnn = DNN(hidden_units_copy, drop_out)
        self.linear = Linear(len(self.dense))
        self.final_linear = nn.Linear(hidden_units_copy[-1], 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedded = [self.embed_layers['embed_' + col_name](sparse_inputs[:, col_num]) for
                           col_name, col_num in
                           zip(self.sparse_fea_num.keys(), range(sparse_inputs.shape[1]))]
        dnn_inputs = torch.cat((torch.cat(sparse_embedded, dim=-1), dense_inputs), dim=-1)
        dnn_outputs = self.dnn(dnn_inputs)
        deep_outputs = self.final_linear(dnn_outputs)

        # wide
        wide_outputs = self.linear(dense_inputs)

        # wide & deep

        return torch.sigmoid(0.5 * (deep_outputs + wide_outputs)).squeeze(-1)
