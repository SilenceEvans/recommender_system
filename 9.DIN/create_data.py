#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：create_data.py
@Author  ：ChenxiWang
@Date    ：2022/9/19 9:30 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    用来存储所有稀疏特征的特征名，对应特征名下的类别总数，对应特征名embedding的维度
    :param feat:
    :param feat_num:
    :param embed_dim:
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    用来存储连续性数值特征
    :param feat:
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, max_len=40):
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)
    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']
    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id'), desc='读取训练数据'):
        pos_list = hist['item_id'].tolist()

        def generate_ng():
            """
            对每个用户进行负采样
            :return:
            """
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [generate_ng() for _ in range(len(pos_list))]
        hist = []  # history_list:记录用户历史行为的列表，即记录用户之前点击过哪些商品
        for i in range(1, len(pos_list)):
            # 每个用户已点击的列表中，最后两个分别用做验证集和测试集
            hist.append(pos_list[i - 1])
            if i == len(pos_list) - 1:
                test_data.append([hist, [pos_list[i]], 1])
                test_data.append([hist, [neg_list[i]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist, [pos_list[i]], 1])
                val_data.append([hist, [neg_list[i]], 0])
            else:
                train_data.append([hist, [pos_list[i]], 1])
                train_data.append([hist, [neg_list[i]], 0])

    # feature columns
    feature_columns = [
        [],
        [sparseFeature('item_id', item_count, embed_dim)]
    ]

    behaviour_list = ['item_id']

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # 数据长度不一致，需要对齐
    # 如果不使用torch中提供的进行padding的工具，像指定用户历史交互序列达到某一指定长度，可以使用下面的代码

    # def concat(df, col):
    #     data = list(df[col])
    #     j = 0
    #     for i in data:
    #         if len(i) < max_len:
    #             pad = [0 for _ in range(max_len - len(i))]
    #             i = i + pad
    #             data.pop(j)
    #             data.insert(j, i)
    #         elif len(i) > max_len:
    #             i = i[:max_len]
    #             data.pop(j)
    #             data.insert(j, i)
    #         j += 1
    #
    #     return data

    # train_X = [np.array([0.] * len(train)), np.array([0] * len(train)), concat(train, 'hist'),
    #            np.array(train['target_item'].tolist())]
    # train_Y = train['label'].values
    #
    # val_X = [np.array([0.] * len(val)), np.array([0] * len(val)), concat(val, 'hist'),
    #          np.array(val['target_item'].tolist())]
    # val_Y = val['label'].values
    #
    # test_X = [np.array([0.] * len(test)), np.array([0] * len(test)), concat(test, 'hist'),
    #           np.array(test['target_item'].tolist())]
    # test_Y = test['label'].values

    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)), list(train['hist']),
               np.array(train['target_item'].tolist())]
    train_Y = train['label'].values

    val_X = [np.array([0.] * len(val)), np.array([0] * len(val)), val['hist'],
             np.array(val['target_item'].tolist())]
    val_Y = val['label'].values

    test_X = [np.array([0.] * len(test)), np.array([0] * len(test)), test['hist'],
              np.array(test['target_item'].tolist())]
    test_Y = test['label'].values

    return feature_columns, behaviour_list, (train_X, train_Y), (val_X, val_Y), (test_X, test_Y)


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence


class DinData(Dataset):
    def __init__(self, data, mode='train'):
        (train_X, train_Y) = data[0]
        (val_X, val_Y) = data[1]
        (test_X, test_Y) = data[2]
        if mode == 'train':
            self.dense_input = train_X[0]  # 连续数值型特征
            self.other_sparse = train_X[1]  # 除去用户交互列表的稀疏特征
            self.input_hist = [torch.LongTensor(i) for i in train_X[2]]  # 用户交互的稀疏特征,目前其中只有id
            self.input_target = train_X[3]  # 需要预测的用户是否点击的id
            self.label = train_Y  # 是否点击标签值
        elif mode == 'val':
            self.dense_input = val_X[0]
            self.other_sparse = val_X[1]
            self.input_hist = [torch.LongTensor(i) for i in val_X[2]]
            self.input_target = val_X[3]
            self.label = val_Y
        else:
            self.dense_input = test_X[0]
            self.other_sparse = test_X[1]
            self.input_hist = [torch.LongTensor(i) for i in test_X[2]]
            self.input_target = test_X[3]
            self.label = test_Y

    def __len__(self):
        return len(self.input_hist)

    def __getitem__(self, idx):
        dense_input = self.dense_input[idx]
        other_sparse = self.other_sparse[idx]
        input_hist = self.input_hist[idx]
        input_target = self.input_target[idx]
        label = self.label[idx]

        return dense_input, other_sparse, input_hist, input_target, label


def collate_fn(batch):
    # 按用户历史序列长短进行排序
    batch = sorted(batch, key=lambda x: len(x[2]), reverse=True)
    need_pad = []
    for i in batch:
        need_pad.append(i[2])
    # 使用torch中提供的pad工具进行padding有一点不好就是不能pading到指定长度，但应该是不影响后续的计算的
    input_hist_pad = pad_sequence(tuple(need_pad), batch_first=True, padding_value=0)
    # print(type(input_hist_pad))
    # print(type(batch))
    j = 0
    for i in batch:
        i = list(i)
        i[2] = input_hist_pad[j, :]
        del batch[j]
        batch.insert(j, tuple(i))
        j += 1
    # print(batch)
    dense_input, other_sparse, input_hist, input_target, label = zip(*batch)
    dense_input = torch.FloatTensor(dense_input).unsqueeze(1)
    other_sparse = torch.LongTensor(other_sparse).unsqueeze(1)
    input_hist = torch.stack(tuple([i for i in input_hist]), dim=0)
    input_target = torch.LongTensor(np.array(input_target))
    label = torch.LongTensor(label).unsqueeze(1)
    return dense_input, other_sparse, input_hist, input_target, label


def get_dataloader(data, mode='train'):
    return DataLoader(DinData(data, mode), batch_size=2, shuffle=False, collate_fn=collate_fn)


if __name__ == '__main__':
    # feature_columns, behaviour_list, (train_X, train_Y), (val_X, val_Y), (
    #     test_X, test_Y) = create_amazon_electronic_dataset('dataset/remap.pkl')
    # print(train_X[2])

    dataloader = get_dataloader()
    batch_data = iter(dataloader).next()
    print(batch_data)
