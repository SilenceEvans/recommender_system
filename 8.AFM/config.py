#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：config.py
@Author  ：ChenxiWang
@Date    ：2022/9/8 4:56 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description : 模型参数配置文件
"""

train_batch_size = 16
test_batch_size = 32

embedding_dim = 10
hidden_units = [269, 128, 64, 32]

dropout = 0.1
output_dim = 1


# AFM相关配置参数
attention_vector = 8
