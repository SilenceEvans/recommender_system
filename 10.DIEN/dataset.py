#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：dataset.py
@Author  ：ChenxiWang
@Date    ：2022/9/27 4:53 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description : 用来处理数据
"""
from random import sample
import torch

import numpy as np
import pandas as pd

# 读取文件
import torch

samples_data = pd.read_csv('data/movie_sample.txt', sep='\t', header=None)
samples_data.columns = ['user_id', 'gender', 'age', 'hist_movie_id', 'hist_len', 'movie_id', 'movie_type_id',
                        'label']

# 数据集
X = samples_data[['user_id', 'gender', 'age', 'hist_movie_id', 'hist_len', 'movie_id', 'movie_type_id']]
y = samples_data['label']


def get_movies_id(data_df):
    """
    获得所有的电影id
    :param data_df:
    :return:
    """
    movies_hist = data_df['hist_movie_id'].values
    movie_id = data_df['movie_id'].values
    all_movies = []
    for hist in movies_hist:
        hist_set = set([int(x) for x in hist.split(',') if x != '0'])
        all_movies.extend(list(hist_set))
    all_movies = list(set(all_movies).union(set(movie_id)))
    return all_movies


def get_neg_click(data_df, neg_num=10):
    """
    定义负采样的方法
    :param data_df: DataFrame结构的数据
    :param neg_num:
    :return:
    """
    movies_set = set(get_movies_id(data_df))
    movies_hist = data_df['hist_movie_id'].values
    neg_movies_list = []
    for hist in movies_hist:
        hist_set = set([int(x) for x in hist.split(',') if x != '0'])

        candidate_neg = sample(list(movies_set - hist_set), neg_num)
        candidate_neg = [str(x) for x in candidate_neg]
        neg_movies_list.append(','.join(candidate_neg))

    return pd.Series(neg_movies_list)


# 进行负采样
X['neg_hist_movies_id'] = get_neg_click(X, neg_num=50)


def get_feature_name(features=X):
    """
    为之后的embedding做准备
    :return:
    """
    dense_fea = ['age', 'hist_len']
    sparse_fea = ['user_id', 'gender', 'hist_movie_id', 'movie_id', 'movie_type_id', 'neg_hist_movies_id']
    sparse_fea_dict = {}
    sparse_fea_dict['user_id'] = features['user_id'].nunique()
    sparse_fea_dict['gender'] = features['gender'].nunique()
    movies_list = get_movies_id(features)
    movie_num = pd.Series(movies_list).nunique() + 1
    sparse_fea_dict['hist_movie_id'] = movie_num
    sparse_fea_dict['movie_id'] = movie_num
    sparse_fea_dict['movie_type_id'] = features['movie_type_id'].nunique()
    sparse_fea_dict['neg_hist_movies_id'] = movie_num
    return dense_fea, sparse_fea, sparse_fea_dict


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DIENDataSet(Dataset):
    def __init__(self):
        super(DIENDataSet, self).__init__()
        data = pd.concat((X, y), axis=1)
        self.inputs = data.drop(columns='label').drop(columns='movie_id').values
        self.targets = data['movie_id'].values
        self.labels = data['label'].values

    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        label = self.labels[idx]
        return input_, target, label

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    # 按用户历史序列长短进行排序
    batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
    need_pad = []
    for i in batch:
        if torch.is_tensor(i[0][3]):
            need_pad.append(i[0][3])
        else:
            need_pad.append([int(x) for x in (i[0][3].split(','))])
            need_pad = [torch.LongTensor(x) for x in need_pad]
    input_hist_pad = pad_sequence(tuple(need_pad), batch_first=True, padding_value=0)
    j = 0
    for i in batch:
        i = list(i)
        i[0][3] = input_hist_pad[j, :]
        del batch[j]
        batch.insert(j, tuple(i))
        j += 1
    input_, target, label = zip(*batch)
    return input_, target, label


def get_dataloader():
    return DataLoader(DIENDataSet(), batch_size=16, shuffle=False, collate_fn=collate_fn)

if __name__ == '__main__':

    pass