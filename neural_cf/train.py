#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：train.py
@Author  ：ChenxiWang
@Date    ：2022/9/6 8:42 下午 
@Github  : https://github.com/
"""
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from ncf import NCF
from ncf_dataset import load_data, get_dataloader
import config

user_num, item_num, train_data, test_data, train_mat = load_data()

model = NCF(user_num=user_num, item_num=item_num, embedding_dim=config.embedding_dim,
            layers_num=config.num_layers, dropout=0.1)
optimizer = Adam(model.parameters())
loss_function = nn.BCEWithLogitsLoss()

loader = get_dataloader()
bar = tqdm(loader, total=len(loader), desc='NCF模型训练')

loss_list = []


def train(epoch):
    for i in range(epoch):
        loader.dataset.ng_sample()
        for idx, (user, item, label) in enumerate(bar):
            optimizer.zero_grad()
            predict = model(user, item)
            label = label.float()
            loss = loss_function(predict, label)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if idx % 10000 == 0:
                torch.save(model.state_dict(), 'models/ncf.pkl')
                torch.save(optimizer.state_dict(), 'models/optimizer.pkl')
                print('epoch:{} idx:{} loss:{:.4f}'.format(epoch, idx, np.mean(loss_list)))


if __name__ == '__main__':
    train(1)
