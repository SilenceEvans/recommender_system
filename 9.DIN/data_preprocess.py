#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：data_preprocess.py
@Author  ：ChenxiWang
@Date    ：2022/9/19 5:19 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description : 数据预处理模块，采用的数据来自亚马逊产品数据集里面的Electronics子集
"""

import numpy as np
import pandas as pd
import pickle
import gc
import random

from tqdm import tqdm

random.seed(2022)


def to_df(file_path):
    """
    将源数据集中的数据转换为dataframe格式
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as f:
        df = {}
        i = 0
        for line in tqdm(f):
            df[i] = eval(line)
            i += 1
            if i > 1000000:
                break
        # 字典df的键要是为dataframe的行名，orient='index
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def generate_pkl():
    """
    将数据缓存起来，之后用
    :return:
    """
    reviews_df = to_df('raw_data/reviews_Electronics_5.json')
    # reviews_df = to_df('raw_data/Electronics.json')

    '''
    reviews_Electronics包含的是用户与商品交互的一些信息，对于文件中的字段描述如下：
        reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
        asin - ID of the product, e.g. 0000013714
        reviewerName - name of the reviewer
        helpful - helpfulness rating of the review, e.g. 2/3
        reviewText - text of the review
        overall - rating of the product
        summary - summary of the review
        unixReviewTime - time of the review (unix time)
        reviewTime - time of the review (raw)
    '''

    with open('raw_data/reviews.pkl', 'wb') as f:
        # pickle.HIGHEST_PROTOCOL
        # Python 3.4以后支持的最高的协议，能够支持很大的数据对象，以及更多的对象类型，并且针对一些数据格式进行了优化。
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

    # 获取所有在reviews_Electronics出现过的商品
    unique_asin = reviews_df['asin'].unique()

    # 释放内存
    del reviews_df
    gc.collect()

    # meta_Electronics包括的是商品的一些元数据信息
    # 处理meta_Electronics  从meta数据集中只保留在reviews文件中出现过的商品
    meta_df = to_df('raw_data/meta_Electronics.json')
    meta_df = meta_df[meta_df['asin'].isin(unique_asin)]

    meta_df = meta_df.reset_index(drop=True)
    pickle.dump(meta_df, open('raw_data/meta.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # print(meta_df.shape)   维度为(28596, 19)


def process_pkl():
    """
    对缓存文件中的信息再次进行删减
    1、reviews_df保留'reviewerID'【用户ID】, 'asin'【产品ID】, 'unixReviewTime'【浏览时间】三列
    2、meta_df保留'asin'【产品ID】, 'categories'【种类】两列
    :return:
    """
    reviews = pd.read_pickle('raw_data/reviews.pkl')
    meta = pd.read_pickle('raw_data/meta.pkl')
    reviews_df = reviews[['reviewerID', 'asin', 'unixReviewTime']]
    # print(reviews_df)
    meta_df = meta[['asin', 'category']]

    del reviews, meta
    gc.collect()

    meta_df['category'] = meta_df['category'].map(lambda x: x[-1])
    # print(meta_df)
    select_user_id = np.random.choice(reviews_df['reviewerID'].unique(), size=100000, replace=False)
    reviews_df = reviews_df[reviews_df['reviewerID'].isin(select_user_id)]
    asin = list(set(list(reviews_df['asin'])) & set(list(meta_df['asin'])))
    # meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    # print(reviews_df.shape, meta_df.shape)
    meta_df = meta_df[meta_df['asin'].isin(asin)]
    reviews_df = reviews_df[reviews_df['asin'].isin(asin)]
    print(len(reviews_df['asin'].unique()), len(meta_df['asin'].unique()))
    return reviews_df, meta_df


def build_map(df, col_name):
    '''
    生成映射，键为列名，值为序列
    :param df:
    :param col_name:
    :return:
    '''
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def save_map():
    reviews_df, meta_df = process_pkl()
    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'category')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')
    user_count, item_count, cate_count, example_count = len(revi_map), len(asin_map), len(cate_map), \
                                                        reviews_df.shape[0]
    # print(user_count, item_count, cate_count, example_count)
    # example_count:总评论个数
    meta_df = meta_df.sort_values('asin').reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime']).reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    cate_list = np.array(meta_df['category'], dtype='int32')

    with open('dataset/remap.pkl', 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, cate_count, example_count), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # generate_pkl()
    # save_info()
    save_map()
    # process_pkl()
