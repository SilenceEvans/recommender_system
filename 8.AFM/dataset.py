#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：dataset.py
@Author  ：ChenxiWang
@Date    ：2022/9/8 4:51 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import pandas as pd
import torch
import config


def get_data():
    train = pd.read_csv('data/processed_data/train_set.csv')
    test = pd.read_csv('data/processed_data/test.csv')
    val = pd.read_csv('data/processed_data/val_set.csv')
    return train, test, val


def get_sparse_fea_num():
    '''
    返回embedding时对应特征需要的类别总数
    :return:
    '''
    train, test, val = get_data()
    all_data = pd.concat((train, val, test))
    sparse = [c for c in all_data.columns if c[0] == 'C']
    dense = [c for c in all_data.columns if c[0] == 'I']
    sparse_fea_num = {}
    for c in sparse:
        sparse_fea_num[c] = all_data[c].nunique()
    return sparse_fea_num, sparse, dense


from torch.utils.data import Dataset, DataLoader


class PNNDataset(Dataset):
    def __init__(self, mode='train'):
        super(PNNDataset, self).__init__()
        if mode == 'train':
            train = pd.read_csv('data/processed_data/train_set.csv')
            self.inputs = train.drop(columns='Label').values
            self.targets = train['Label'].values
        elif mode == 'val':
            val = pd.read_csv('data/processed_data/val_set.csv')
            self.inputs = val.drop(columns='Label').values
            self.targets = val['Label'].values
        else:
            test = pd.read_csv('data/processed_data/test.csv')
            self.inputs = test.values

    def __getitem__(self, idx):

        input_ = self.inputs[idx]
        target = self.targets[idx]

        return input_, target

    def __len__(self):
        return len(self.inputs)


def get_dataloader(mode='train'):
    if mode in ('train', 'val'):
        batch_size = config.train_batch_size
    else:
        batch_size = config.test_batch_size
    return DataLoader(PNNDataset(mode), shuffle=True, batch_size=batch_size)


if __name__ == '__main__':
    loader = get_dataloader()
    for idx, (input_, target) in enumerate(loader):
        print(idx, '\n')
        print(input_, '\n')
        print(target, '\n')
        break
