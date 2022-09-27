#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：pnn_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/10 5:37 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloader
from dataset import get_sparse_fea_num
from pnn import config


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.float()
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class ProductLayer(nn.Module):

    def __init__(self, mode, embed_dim, field_num, hidden_units):
        '''
        这块没有用到网上那些代码中的w矩阵，而是用torch中的线性层替代了，因为我感觉原理是一样的，只要这几个矩阵维度能够倒清楚，
        用了参数矩阵的做法反而感觉抽象
        :param mode:
        :param embed_dim:
        :param field_num:
        :param hidden_units:
        '''

        super(ProductLayer, self).__init__()
        self.mode = mode
        self.lz = nn.Linear(embed_dim * field_num, hidden_units[0])
        self.lp_in = nn.Linear(field_num * field_num, hidden_units[0])
        self.lp_out = nn.Linear(embed_dim * embed_dim, hidden_units[0])
        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)

    def forward(self, x):
        '''

        :param x: embedding之后的sparse特征[batch_size,filed_num,embedding_dim]
        :return:
        '''
        lz = self.lz(x.reshape((x.shape[0], -1)))
        if self.mode == 'in':
            lp_input = x.matmul(x.transpose(2, 1)).reshape(x.shape[0], -1)
            lp = self.lp_in(lp_input)
        else:
            lp_input = torch.unsqueeze(torch.sum(x, dim=1), dim=1)
            lp_input_T = lp_input.transpose(2, 1)
            lp_input = lp_input_T.matmul(lp_input).reshape(lp_input.shape[0], -1)
            lp = self.lp_out(lp_input)

        return lz + lp + self.l_b


from dataset import get_sparse_fea_num


class PNN(nn.Module):
    def __init__(self, mode, embedding_dim=config.embedding_dim, hidden_units=config.hidden_units,
                 dnn_dropout=config.dropout):
        super(PNN, self).__init__()
        self.sparse_fea_num, self.sparse, self.dense = get_sparse_fea_num()
        self.embedding_dim = embedding_dim
        self.filed_num = len(self.sparse)
        self.dense_num = len(self.dense)
        self.mode = mode
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + str(key): nn.Embedding(val, self.embedding_dim) for key, val in
             self.sparse_fea_num.items()})
        self.product = ProductLayer(mode, self.embedding_dim, self.filed_num, hidden_units)
        # todo 不进行下面这一步将来保存模型重新加载的时候会报错 error type:size mismatch，而且dnn中的第一个线性层的参数也会变成269而不是256
        hidden_units_copy = hidden_units.copy()
        hidden_units_copy[0] += self.dense_num
        self.dnn_net = DNN(hidden_units_copy, dnn_dropout)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedded = [self.embedding_layers['embedding_' + str(key)](sparse_inputs[:, x])
                           for key, x in zip(self.sparse_fea_num.keys(), range(sparse_inputs.shape[1]))]

        # sparse_embedded[26,16,10]
        sparse_embedded = torch.stack(sparse_embedded)
        sparse_embedded = sparse_embedded.transpose(1, 0)
        i = sparse_embedded
        pro_output = self.product(i)
        cat = torch.cat((pro_output, dense_inputs), dim=-1)
        dnn_output = self.dnn_net(cat)
        output = self.final_linear(dnn_output)
        return torch.sigmoid(output.squeeze(1))


if __name__ == '__main__':
    loader = get_dataloader()

    net = PNN(mode='out')
    for idx, (inputs, labels) in enumerate(loader):
        predictions = net(inputs)
        print(predictions.shape)
        break
