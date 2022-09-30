#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：dien_model.py
@Author  ：ChenxiWang
@Date    ：2022/9/27 5:58 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import numpy as np
import torch
import torch.nn as nn
from dataset import get_feature_name
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F


class LocalActivationUnit(nn.Module):
    def __init__(self, attn_hidden_units):
        super(LocalActivationUnit, self).__init__()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in zip(attn_hidden_units[:-1], attn_hidden_units[1:])])
        self.attn_final_linear = nn.Linear(attn_hidden_units[-1], 1)

    def forward(self, *inputs):
        # k=v=gru_outputs,[batch_size,hist_len,embedding_dim]   q=targets,[batch_size,1,embedding_dim]
        k, v, q = inputs
        q = q.repeat(1, k.size()[1], 1)
        linear_input = torch.cat([q, k, q - k, q * k], dim=-1)
        for linear in self.linear_layers:
            linear_input = linear(linear_input)
        linear_output = self.attn_final_linear(linear_input)
        mask = (linear_output == 0)
        score = F.softmax(linear_output.masked_fill(mask, -np.inf), dim=0)
        # score:[batch_size,length,1]
        return score


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )
        self.candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, inputs, hidden, attn_score):
        """

        :param inputs:
        :param hidden:
        :param attn_score: [batch_size,length,1]
        :return:
        """
        u = self.update_gate(torch.hstack([inputs, hidden]))  # [batch_size,length,1]
        u = attn_score * u
        r = self.reset_gate(torch.hstack([inputs, hidden]))
        tilde_h = self.candidate(torch.hstack([inputs, hidden * r]))
        h = (1 - u) * hidden + u * tilde_h
        return h


class AUGRU(nn.Module):
    """
    兴趣进化层
    """

    def __init__(self, attn_hidden_units, embedding_size):
        super(AUGRU, self).__init__()
        self.attention = LocalActivationUnit(attn_hidden_units)
        self.hidden_size = embedding_size
        self.input_size = embedding_size
        self.gru_cell = GRUCell(self.input_size, self.hidden_size)

    def forward(self, *inputs):
        gru_outputs, targets = inputs
        scores = self.attention(gru_outputs, gru_outputs, targets)
        hidden = torch.zeros((gru_outputs.size()[0], self.hidden_size))
        # output = torch.zeros((gru_outputs.size()[0], gru_outputs.size()[1], self.hidden_size))
        # start = 0
        for i in range(gru_outputs.size(1)):
            inputs = gru_outputs[:, i, :]
            hidden = self.gru_cell(inputs, hidden, scores[:,i,:])
        return hidden


class DIEN(nn.Module):
    def __init__(self, max_length=None, embedding_dim=8, attn_hidden_units=(32, 80, 40)):
        super(DIEN, self).__init__()
        dense_fea, sparse_fea, sparse_fea_dict = get_feature_name()
        self.embedding_layers = nn.ModuleDict(
            {'embedding_' + key: nn.Embedding(value, embedding_dim, padding_idx=0) for key, value in
             sparse_fea_dict.items()})
        self.gru = nn.GRU(embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.augru = AUGRU(attn_hidden_units, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 200),
            nn.Linear(200, 80),
            nn.Linear(80, 1)
        )

    def forward(self, inputs, targets):
        user_id, gender, age, hist_movie_id, hist_len, movie_type_id, neg_hist_movies_id = zip(*inputs)
        user_id = torch.LongTensor(user_id).unsqueeze(-1)
        gender = torch.LongTensor(gender).unsqueeze(-1)
        hist_len = torch.LongTensor(hist_len)
        hist_movie_id = torch.stack(hist_movie_id, dim=0)
        movie_type_id = torch.LongTensor(movie_type_id)

        targets = torch.LongTensor(targets).unsqueeze(-1)

        # 进行embedding
        hist_movie_id_embedding = self.embedding_layers['embedding_hist_movie_id'](hist_movie_id)

        targets_embedding = self.embedding_layers['embedding_movie_id'](targets)
        # 先经过抽取兴趣特征的GRU
        # 将padding过的数据进行压缩
        hist_len = (hist_movie_id[:, :-1] > 0).sum(dim=1)
        input_packed = pack_padded_sequence(hist_movie_id_embedding, lengths=hist_len, batch_first=True,enforce_sorted=False)
        # try:
        packed_gru_output, hidden = self.gru(input_packed)
        # except:
        # 将得到的输出再解压，得到能与target对齐的效果，即一组gru输出的数据对应一个target
        output_padded, hist_len = pad_packed_sequence(packed_gru_output, batch_first=True)
        augru_output = self.augru(output_padded, targets_embedding)
        # user_id_embedding = self.embedding_layers['embedding_user_id'](user_id)
        mlp_input = torch.cat((augru_output, targets_embedding.squeeze(1)), dim=-1)
        output = self.mlp(mlp_input)
        result = torch.sigmoid(output).squeeze(-1)
        return result



