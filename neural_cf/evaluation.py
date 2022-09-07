#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：evaluation.py
@Author  ：ChenxiWang
@Date    ：2022/9/6 10:06 下午 
@Github  : https://github.com/
"""
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ncf import NCF
from neural_cf.ncf_dataset import load_data, get_dataloader
import config

user_num, item_num, train_data, test_data, train_mat = load_data()
ncf = NCF(user_num=user_num, item_num=item_num, embedding_dim=config.embedding_dim,
          layers_num=config.num_layers, dropout=0.1)
if os.path.exists('models/ncf.pkl'):
    ncf.load_state_dict(torch.load('models/ncf.pkl'))

# 评价指标用NDCG
test_loader = get_dataloader(train=False)
bar = tqdm(test_loader, total=len(test_loader), desc='测试ncf')


def hit(get_items, pred_items):
    i = 0
    for pred_item in pred_items:
        if pred_item in get_items:
            i += 1
    return i


def ndcg(get_items, pred_items):
    ndcg = 0
    for pred_item in pred_items:
        if pred_item in get_items:
            idx = pred_items.index(pred_item)
            ndcg += np.reciprocal(np.log2(idx + 2))
    return ndcg


def eval():
    HR, NDCG = [], []
    for idx, (user, item, label) in enumerate(bar):
        with torch.no_grad():
            predict = ncf(user, item)
            _, indices = torch.topk(predict, config.topk)
            # 模型的得到的输出结果的索引和item的索引是一样的，即模型输出的索引对应item的索引
            recommends = torch.take(item, indices).numpy().tolist()
            HR.append(hit(item.numpy(), recommends))
            NDCG.append(ndcg(item.numpy(), recommends))
    return np.mean(HR), np.mean(NDCG)


if __name__ == '__main__':
    print(eval())
