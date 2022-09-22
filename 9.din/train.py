#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：train.py
@Author  ：ChenxiWang
@Date    ：2022/9/21 8:04 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
from create_data import get_dataloader, create_amazon_electronic_dataset
from din import DIN
from tqdm import tqdm

feature_columns, behaviour_list, (train_X, train_Y), (val_X, val_Y), (
    test_X, test_Y) = create_amazon_electronic_dataset('./dataset/remap.pkl')
data = [(train_X, train_Y), (val_X, val_Y), (test_X, test_Y)]
dataloader = get_dataloader(data)
din = DIN(feature_columns, behaviour_list)
for idx, (dense_input, other_sparse, input_hist, input_target, label) in enumerate(dataloader):
    predicts = din(dense_input, other_sparse, input_hist, input_target)
    break
