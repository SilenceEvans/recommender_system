#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：din.py
@Author  ：ChenxiWang
@Date    ：2022/9/20 3:44 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):

    def __init__(self, att_hidden_units):
        super(AttentionLayer, self).__init__()
        self.dense = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in zip(att_hidden_units[:-1], att_hidden_units[1:])])
        self.att_final_linear = nn.Linear(att_hidden_units[-1], 1)

    def forward(self, x):
        """
        item_embedded:[batch_size,1,embedding_dim]
        hist_embedded:batch_size,max_length,embedding_dim]
        :param x: (item_embedded,hist_embedded,hist_embedded，mask)-->(q,k,v,mask)
        :return:
        """
        q, k, v, mask = x
        q = q.repeat(1, k.size(1), 1)
        q = q.reshape(-1, k.size(1), k.size(2))
        # 作者说这里是有利于模型相关性建模的显性知识，有的还加上差的，就类似于PNN里面的(concat(A, B, A-B, A*B)。
        info = torch.cat([q, k, q - k, q * k], dim=-1)
        for linear in self.dense:
            info = linear(info)
        info = torch.sigmoid(info)

        info_final_linear = self.att_final_linear(info)
        # 用mask对之前padding位置处的元素进行极小负值替换
        score = F.softmax(info_final_linear.masked_fill(mask, -np.inf), dim=1)
        score = score.transpose(2, 1)
        # score[batch_size,1,max_length]
        # score * v
        results = torch.matmul(score, v)
        outputs = results.squeeze(1)
        return outputs


class DIN(nn.Module):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(32, 80, 40),
                 ffn_hidden_units=(9, 80, 40), att_activation='sigmoid',
                 ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        super(DIN, self).__init__()
        self.maxlen = maxlen
        self.behavior_feature_list = behavior_feature_list
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        self.behaviour_embedding_layers = nn.ModuleDict(
            {'embedding_' + str(i['feat']): nn.Embedding(i['feat_num'], i['embed_dim']) for i in
             self.sparse_feature_columns if i['feat'] in self.behavior_feature_list})

        self.other_sparse_embedding_layers = nn.ModuleDict(
            {'embedding_' + str(i['feat']): nn.Embedding(i['feat_num'], i['embed_dim']) for i in
             self.sparse_feature_columns if
             i['feat'] not in self.behavior_feature_list})

        self.attention_layer = AttentionLayer(att_hidden_units)

        self.bn = nn.BatchNorm1d(
            (1 + self.behavior_num * 8))  # 这儿直接写死了，有需要的话可以将embedding_dim定义成变量

        # 定义DNN全连接网络
        self.dnn = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in zip(ffn_hidden_units[:-1], ffn_hidden_units[1:])])

        self.drop_out = nn.Dropout(p=0.2)
        self.final_linear = nn.Linear(ffn_hidden_units[-1], 1)

    def forward(self, *inputs):
        """
        *inputs包括dense_inputs, other_sparse_inputs, hist_inputs, item_inputs，其中：
        dense_inputs：连续数值型特征，size：[batch_size,1]
        other_sparse_inputs：未列入用户交互的类别型特征，size：[batch_size,1]
        hist_inputs：用户交互的类别型特征，size：[batch_size,特征数]
        item_inputs：需要预测用户是否点击的商品id,size[batch_size,1]
        :param inputs:
        :return:
        """
        dense_inputs, other_sparse_inputs, hist_inputs, item_inputs = inputs
        other_sparse_inputs = [
            self.other_sparse_embedding_layers['embedding_' + str(i['feat'])](other_sparse_inputs)
            for i in
            self.sparse_feature_columns if i['feat'] not in self.behavior_feature_list]
        # other_sparse_inputs = torch.cat(other_sparse_inputs,dim=-1)
        # other_info = torch.cat((dense_inputs,other_sparse_inputs),dim=-1)
        # 因为目前other_sparse_inputs没有任何东西，所以直接略去这一项
        other_info = dense_inputs

        # 定义mask，之后在attention中softmax时，padding位置处的值不进行计算
        mask = (hist_inputs == 0).unsqueeze(-1)
        hist_inputs = [self.behaviour_embedding_layers['embedding_' + str(i['feat'])](hist_inputs) for i in
                       self.sparse_feature_columns if i['feat'] in self.behavior_feature_list]
        hist_inputs = hist_inputs[0]

        item_inputs = [self.behaviour_embedding_layers['embedding_' + str(i['feat'])](item_inputs) for i in
                       self.sparse_feature_columns if i['feat'] in self.behavior_feature_list]
        # item_inputs[batch_size,1,embedding_dim]
        item_inputs = item_inputs[0]

        # 下面进行attention计算
        attn_out = self.attention_layer(
            [item_inputs, hist_inputs, hist_inputs, mask])  # attn_out:[batch_size,embedding_dim]

        # 将所有特征拼接起来
        all_feature = torch.cat((other_info, attn_out), dim=-1)

        bn_out = self.bn(all_feature)

        for linear in self.dnn:
            bn_out = linear(bn_out)

        final_linear_out = self.final_linear(self.drop_out(bn_out))

        return torch.sigmoid(final_linear_out)
