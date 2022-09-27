#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：afm_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/18 8:28 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from dataset import get_sparse_fea_num


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in zip(hidden_units[:-1], hidden_units[1:])])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.float()
        for linear in self.dnn_network:
            x = F.relu(linear(x))
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, *att_args):
        '''
        :param *att_args: 这里做一个闭包的处理，其中的参数就是embedding_dim,att_vector
        '''
        super(Attention, self).__init__()
        self.att_w = nn.Linear(att_args[0], att_args[1])  # 先对其进行一个线性变换
        self.att_weight_ = nn.Linear(att_args[1], 1)  # 求注意力权重前的线性变换

    def forward(self, x):
        w_processed = self.att_w(x)
        att_weights_ = self.att_weight_(w_processed)
        att_weights = F.softmax(att_weights_, dim=1)
        out = torch.sum(att_weights * x, dim=1)

        return out


class AFM(nn.Module):
    def __init__(self, hidden_units=config.hidden_units, embedding_dim=config.embedding_dim,
                 dropout=config.dropout, att_vector=config.attention_vector):
        super(AFM, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        dnn_input_dim = embedding_dim + len(self.dense)
        hidden_units_copy = hidden_units.copy()
        hidden_units_copy.insert(0, dnn_input_dim)
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + str(key): nn.Embedding(val, embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        self.dnn = DNN(hidden_units_copy, dropout)
        self.attention = Attention(embedding_dim, att_vector)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedded = [self.embedding_layers['embedding_' + str(fea_name)](sparse_inputs[:,fea]) for
                           fea_name, fea in zip(self.sparse, range(sparse_inputs.shape[1]))]
        sparse_embedded = torch.stack(sparse_embedded).transpose(1, 0)

        # 做attention层需要的特征之间元素积的操作
        first = []
        second = []

        for f, s in itertools.combinations(range(sparse_embedded.shape[1]), 2):
            # itertools.combinations 生成的是组合的一个个元组，元组的数量是等差数列求和个，即：n*(n-1) / 2
            first.append(f)
            second.append(s)

        p = sparse_embedded[:, first, :]
        q = sparse_embedded[:, second, :]

        # 做两两对应元素积操作
        bi_interaction = p * q
        att_out = self.attention(bi_interaction)

        dnn_input = torch.cat((dense_inputs, att_out), dim=-1)
        dnn_output = self.dnn(dnn_input)
        output = self.final_linear(dnn_output)
        return torch.sigmoid(output).squeeze(-1)
