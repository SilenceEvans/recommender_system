#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：nfm_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/16 8:25 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch.nn as nn
import torch
from dataset import get_sparse_fea_num
import config
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in zip(hidden_units[:-1], hidden_units[1:])])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        return self.dropout(x)


class NFM(nn.Module):
    def __init__(self, hidden_units=config.hidden_units, dropout=config.dropout,
                 embedding_dim=config.embedding_dim):
        super(NFM, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + str(key): nn.Embedding(val, embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        hidden_units_copy = hidden_units.copy()
        dnn_inputs_dim = len(self.dense) + embedding_dim
        hidden_units_copy.insert(0, dnn_inputs_dim)
        self.bn = nn.BatchNorm1d(dnn_inputs_dim)
        self.dnn = DNN(hidden_units_copy, dropout)
        self.nn_final_linear = nn.Linear(hidden_units[-1],1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedded = [self.embedding_layers['embedding_' + str(key)](sparse_inputs[:, val]) for key, val
                           in zip(self.sparse, range(sparse_inputs.shape[1]))]
        sparse_embedded = torch.stack(sparse_embedded).transpose(1, 0)
        # a = torch.pow(torch.sum(sparse_embedded, dim=1), 2)
        # b = torch.sum(torch.pow(sparse_embedded, 2), dim=1)
        embedded_cross = 1 / 2 * (torch.pow(torch.sum(sparse_embedded, dim=1), 2) - torch.sum(
            torch.pow(sparse_embedded, 2), dim=1))

        dnn_inputs = torch.cat((dense_inputs,embedded_cross),dim=-1).float()
        dnn_inputs = self.bn(dnn_inputs)
        dnn_outputs = self.dnn(dnn_inputs)
        outputs = self.nn_final_linear(dnn_outputs)

        outputs = torch.sigmoid(outputs).squeeze(-1)

        return outputs
