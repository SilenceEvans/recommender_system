#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：process_data.py
@Author  ：ChenxiWang
@Date    ：2022/9/8 3:50 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :前期数据清洗
"""
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

label = train_df['Label']

# 数据合并
del train_df['Label']

data_df = pd.concat((train_df, test_df))
del data_df['Id']

# print(data_df.columns,data_df.shape)

# 将数值特征与类别特征进行区分
dense_fea = [c for c in data_df.columns if c[0] == 'I']
sparse_fea = [c for c in data_df.columns if c[0] == 'C']
# print('dense:',dense_fea,'\nsparse:',sparse_fea)

# 填充缺失值
data_df[dense_fea] = data_df[dense_fea].fillna(0)
data_df[sparse_fea] = data_df[sparse_fea].fillna('-1')

# 将数值特征进行归一化处理(0,1)之间，类别特征进行编码处理
for f in sparse_fea:
    le = LabelEncoder()
    # print(data_df[f])
    data_df[f] = le.fit_transform(data_df[f])

mms = MinMaxScaler()
data_df[dense_fea] = mms.fit_transform(data_df[dense_fea])

# print(data_df[dense_fea])
# print(data_df[sparse_fea])

# 分开训练集与测试集
train = data_df[:train_df.shape[0]]
test = data_df[train_df.shape[0]:]

train['Label'] = label

# 划分数据集

train_set, val_set = train_test_split(train, test_size=0.2, random_state=3)
# drop=True：删除原来索引,建立新索引  inplace=True:不创建新的dataframe对象，直接在原有的基础上进行修改
train_set.reset_index(drop=True, inplace=True)
val_set.reset_index(drop=True, inplace=True)

if not os.path.exists('data/processed_data/'):
    os.makedirs('data/processed_data/')
# index=0，删除行索引
train_set.to_csv('data/processed_data/train_set.csv', index=0)
val_set.to_csv('data/processed_data/val_set.csv', index=0)
test.to_csv('data/processed_data/test.csv', index=0)



