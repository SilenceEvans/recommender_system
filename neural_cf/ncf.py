#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models
@File    ：ncf.py
@Author  ：ChenxiWang
@Date    ：2022/9/6 7:29 下午
@Github  : https://github.com/
"""
import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    组合gmf和mlp
    """

    def __init__(self, user_num, item_num, embedding_dim, layers_num, dropout):
        super(NCF, self).__init__()
        # gmf模块
        self.gmf_user_embedding = nn.Embedding(user_num, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(item_num, embedding_dim)

        # mlp模块
        self.mlp_user_embedding = nn.Embedding(user_num, embedding_dim * (2 ** (layers_num - 1)))
        self.mlp_item_embedding = nn.Embedding(item_num, embedding_dim * (2 ** (layers_num - 1)))

        MLP_Modules = []

        for i in range(layers_num):
            input_size = embedding_dim * (2 ** (layers_num - i))
            MLP_Modules.append(nn.Dropout(p=dropout))
            MLP_Modules.append(nn.Linear(input_size, input_size // 2))
            MLP_Modules.append(nn.ReLU())

        self.MLP_layers = nn.Sequential(*MLP_Modules)

        self.predict_layer = nn.Linear(2 * embedding_dim, 1)

    def _init_weight_(self):
        nn.init.normal_(self.gmf_user_embedding.weight)
        nn.init.normal_(self.gmf_item_embedding.weight)
        nn.init.normal_(self.mlp_user_embedding.weight)
        nn.init.normal_(self.mlp_item_embedding.weight)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):
        gmf_user_embedded = self.gmf_user_embedding(user)
        gmf_item_embedded = self.gmf_item_embedding(item)
        gmf_hadamard_product = gmf_user_embedded.mul(gmf_item_embedded)

        mlp_user_embedded = self.mlp_user_embedding(user)
        mlp_item_embedded = self.mlp_item_embedding(item)
        mlp_concatenation = torch.cat((mlp_user_embedded, mlp_item_embedded), dim=-1)

        mlp_output = self.MLP_layers(mlp_concatenation)

        ncf_concatenation = torch.cat((gmf_hadamard_product, mlp_output),dim=-1)
        ncf_output = self.predict_layer(ncf_concatenation)

        return ncf_output.view(-1)
