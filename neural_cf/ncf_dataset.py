#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：ncf_dataset.py
@Author  ：ChenxiWang
@Date    ：2022/9/6 5:29 下午 
@Github  : https://github.com/
"""
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from neural_cf import config


def load_data():
    # read_csv,usecols指定读取哪几列的数据
    train_data = pd.read_csv('data/ml-1m.train.rating', sep='\t', header=None, names=['user', 'item'], usecols=[0, 1],
                             dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # 构建一个稀疏的评分矩阵
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []

    with open('data/ml-1m.test.negative', 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            data = line.split('\t')
            user = eval(data[0])[0]  # 将字符串'(a,1)'转换成元组(a,b)
            item = eval(data[0])[1]

            test_data.append([user, item])

            for i in data[1:]:
                test_data.append([user, int(i)])
            line = fd.readline()
    return user_num, item_num, train_data, test_data, train_mat


class NCFDataSet(Dataset):

    def __init__(self, features, item_num, train_mat=None, num_ng=0, is_training=None):
        super(NCFDataSet, self).__init__()
        self.is_training = is_training
        self.num_ng = num_ng
        self.train_mat = train_mat
        self.item_num = item_num
        self.features_ps = features
        self.labels = [0 for _ in range(len(self.features_ps))]

    def ng_sample(self):
        """
        主要用来处理在训练数据中加入负采样数据
        :return:
        """
        assert self.is_training, 'no need to sapling when training'

        self.features_ng = []
        # 每一位用户加入4个负采样样本
        for x in self.features_ps:
            user = x[0]
            for i in range(self.num_ng):
                item = np.random.randint(self.item_num)
                while (user, item) in self.train_mat:
                    item = np.random.randint(self.item_num)
                self.features_ng.append([user, item])
        self.labels_ps = [1 for _ in range(len(self.features_ps))]
        self.labels_ng = [0 for _ in range(len(self.features_ng))]

        # 加了负采样样本之后的全部特征和目标值
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = self.labels_ps + self.labels_ng

    def __getitem__(self, idx):
        """
        如果是训练，则样本选用的是用户已经点击过或者是评过分的
        如果是测试，样本全部选择已经评过分的，只不过将此时的label全部设为0
        :param idx:
        :return:
        """

        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]

        label = labels[idx]

        return user, item, label

    def __len__(self):

        return len(self.labels) * (1 + self.num_ng)


def get_dataloader(train=True):
    user_num, item_num, train_data, test_data, train_mat = load_data()
    if train:
        batch_size = config.train_batch_size
        dataset = NCFDataSet(features=train_data, item_num=item_num, train_mat=train_mat, num_ng=config.num_ng_train,
                             is_training=train)
    else:
        batch_size = config.test_batch_size
        dataset = NCFDataSet(features=test_data, item_num=item_num, num_ng=config.num_ng_test, is_training=train)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)



