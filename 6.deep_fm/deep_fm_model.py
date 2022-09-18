#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：deep_fm_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/16 2:50 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch
import torch.nn as nn
import config
from dataset import get_sparse_fea_num
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, w_latent_dim, fea_num):
        super(FM, self).__init__()
        self.w0 = nn.Parameter(torch.zeros([1]))
        self.w1 = nn.Parameter(torch.rand([fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([fea_num, w_latent_dim]))

    def forward(self, inputs):
        inputs = inputs.float()
        first_order = self.w0 + torch.mm(inputs, self.w1)
        second_order = 1/2 * torch.sum(
            torch.pow(torch.mm(inputs, self.w2), 2) - torch.mm(torch.pow(inputs, 2), torch.pow(self.w2, 2)),
            dim=-1,keepdim=True)
        return first_order + second_order


class DNN(nn.Module):
    def __init__(self, hidden_units, drop_out):
        super(DNN, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])

        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = F.relu(x)
        return self.dropout(x)


class DeepFM(nn.Module):
    def __init__(self, embedding_dim=config.embedding_dim, hidden_units=config.hidden_units,
                 drop_out=config.dropout):
        super(DeepFM, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + str(key): nn.Embedding(val, embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        hidden_units_copy = hidden_units.copy()
        inputs_dim = len(self.sparse) * embedding_dim + len(self.dense)
        hidden_units_copy.insert(0, inputs_dim)
        self.dnn = DNN(hidden_units_copy, drop_out)
        self.fm = FM(embedding_dim, inputs_dim)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, inputs):
        dense_features, sparse_features = inputs[:, :13], inputs[:, 13:]
        sparse_features = sparse_features.long()
        sparse_embedded = [self.embedding_layers['embedding_' + str(key)](sparse_features[:, num]) for
                           key, num in
                           zip(self.sparse, range(sparse_features.shape[1]))]
        sparse_embedded = torch.cat(sparse_embedded, dim=-1)
        deep_fm_inputs = torch.cat((dense_features, sparse_embedded), dim=-1).float()

        # fm
        fm_result = self.fm(deep_fm_inputs)
        # dnn
        dnn_result = self.final_linear(self.dnn(deep_fm_inputs))
        outputs = torch.add(fm_result,dnn_result)

        return torch.sigmoid(outputs).squeeze(-1)
