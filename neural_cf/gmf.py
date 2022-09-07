#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：gmf.py
@Author  ：ChenxiWang
@Date    ：2022/9/6 5:28 下午 
@Github  : https://github.com/
"""

from torch import nn


class GMF(nn.Module):

    def __init__(self,user_num,item_num,embedding_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(user_num,embedding_dim)
        self.item_embedding = nn.Embedding(item_num,embedding_dim)

        self.linear = nn.Linear(embedding_dim,1)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight,std=0.01)
        nn.init.normal_(self.item_embedding.weight,std = 0.01)

    def forward(self,user,item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        hadamard_product = user_embedded.mul(item_embedded)
        output = self.linear(hadamard_product)

        return output.view(-1)
