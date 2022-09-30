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
import pandas as pd
import torch
from matplotlib import pyplot as plt

from dataset import get_dataloader
from dien_model import DIEN
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import auc, roc_auc_score, roc_curve


def auc(y_pred, y_true):
    y_pred = y_pred.data
    y = y_true.data

    return roc_auc_score(y, y_pred)


dataloader = get_dataloader()
dien = DIEN()
optimizer = Adam(dien.parameters())
loss_function = nn.BCELoss()

history = pd.DataFrame(columns=['epoch', 'loss', 'auc', 'val_loss', 'val_auc'])


def train(epoch):
    loss_list = []
    metric_list = []
    bar = tqdm(dataloader, total=len(dataloader), desc='DIN Model training,epoch{}'.format(epoch))
    for idx, (input_, target, label) in enumerate(bar):
        optimizer.zero_grad()
        predicts = dien(input_, target)
        label = torch.FloatTensor(label)
        loss = loss_function(predicts, label)
        try:
            metric_list.append(auc(predicts, label).item())
        except:
            pass
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            torch.save(dien.state_dict(), 'models/dien.pkl')
            torch.save(optimizer.state_dict(), 'models/optimizer.pkl')
    # print('epoch:{}  loss:{:.4f}'.format(epoch, np.mean(loss_list)))
    history.loc[epoch, ['epoch', 'loss', 'auc']] = epoch, np.mean(loss_list), np.mean(metric_list)


# def eval(epoch):
#     eval_loader = get_dataloader(data, mode='val')
#     loss_list = []
#     metric_list = []
#     din.load_state_dict(torch.load('models/din.pkl'))
#     with torch.no_grad():
#         bar = tqdm(dataloader, total=len(eval_loader),desc='DIN Model eval')
#         for idx, (dense_input, other_sparse, input_hist, input_target, label) in enumerate(bar):
#             predicts = din(dense_input, other_sparse, input_hist, input_target)
#             loss = loss_function(predicts, label.float())
#             metric_list.append(auc(predicts, label).item())
#             loss_list.append(loss.item())
#         print('epoch:{}  loss:{:.4f}'.format(epoch, np.mean(loss_list)))
#         history.loc[epoch, ['epoch', 'val_loss', 'val_auc']] = epoch, np.mean(loss_list), np.mean(metric_list)


def plot_metric(metric_name):
    train_metric = history[metric_name]
    val_metric = history['val_' + metric_name]
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'bo--')
    plt.plot(epochs, val_metric, 'ro--')
    plt.title('DIN training and validation_' + metric_name)
    plt.xlabel('epochs')
    plt.ylabel(metric_name)
    plt.legend(['train_' + metric_name, 'val_' + metric_name])
    plt.show()


if __name__ == '__main__':
    for i in range(100):
        train(i)
        # eval(i)
    plot_metric('loss')
    plot_metric('auc')
