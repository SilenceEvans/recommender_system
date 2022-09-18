#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：deep_crossing_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/8 4:40 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from dataset import get_dataloader, get_sparse_fea_num

import config


class ResidualBlock(nn.Module):
    def __init__(self, dim_stack, hidden_stack):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_stack)
        self.linear2 = nn.Linear(hidden_stack, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orig_x = x.clone()
        outputs = self.relu(orig_x + self.linear2(self.linear1(x)))
        return outputs


class DeepCrossing(nn.Module):
    def __init__(self, embedding_dim=config.embedding_dim, hidden_units=config.hidden_units,
                 dropout=config.dropout, output_dim=config.output_dim):
        super(DeepCrossing, self).__init__()
        self.sparse_fea_num, sparse, dense = get_sparse_fea_num()

        self.embedded_layers = nn.ModuleDict(
            {'embed_' + str(col): nn.Embedding(num, embedding_dim) for col, num in
             self.sparse_fea_num.items()})

        embedd_dim_sum = len(sparse) * config.embedding_dim
        dim_stack = len(dense) + embedd_dim_sum

        self.res_layers = nn.ModuleList(
            [ResidualBlock(dim_stack, hidden_unit) for hidden_unit in hidden_units])

        self.res_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(dim_stack, output_dim)

    def forward(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        dense_inputs = dense_inputs.float()
        sparse_inputs = sparse_inputs.long()
        # dense_inputs(batch_size,13),sparse_inputs(batch_size,26)
        sparse_embeds = []
        for k, i in zip(self.sparse_fea_num.keys(), range(sparse_inputs.shape[1])):
            sparse_embeds.append(self.embedded_layers['embed_' + k](sparse_inputs[:, i]))
        # sparse_embeds(batch_size,26,embedding_size)
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        # sparse_embeds = torch.tensor(sparse_embeds).view(sparse_inputs.shape[0], -1)
        stack = torch.cat((dense_inputs, sparse_embeds), dim=-1)

        for res in self.res_layers:
            stack = res(stack)

        r = stack
        r = self.res_dropout(r)
        outputs = torch.sigmoid(self.linear(r))
        return outputs.squeeze(1)


if __name__ == '__main__':
    dc_model = DeepCrossing()
    loader = get_dataloader()
    for idx, (inputs, targets) in enumerate(loader):
        outputs = dc_model(inputs)
        break
