#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rec_models 
@File    ：train.py
@Author  ：ChenxiWang
@Date    ：2022/9/8 9:10 下午 
@Github  : https://github.com/SilenceEvans/recommender_system
@Description :
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm
from dataset import get_dataloader
from pnn_model import PNN
from torch.optim import Adam
import pandas as pd


def auc(y_pred, y_true):
    y_pred = y_pred.data
    y = y_true.data

    return roc_auc_score(y, y_pred)


metric_func = auc
metric_name = 'auc'

history = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', ('val_' + metric_name)])


def train(epoch):
    model = PNN(mode='out')
    optimizer = Adam(model.parameters(),lr=0.001)
    loss_function = nn.BCELoss()
    loss_list = []
    metric_list = []
    loader = get_dataloader()
    bar = tqdm(loader,total=len(loader),desc='dc训练')
    for idx, (inputs, labels) in enumerate(bar):
        # print(labels)
        optimizer.zero_grad()
        predicts = model(inputs)
        loss = loss_function(predicts, labels.float())
        loss_list.append(loss.item())
        try:
            metric_list.append(metric_func(predicts, labels).item())
        except ValueError:
            # print('二元分类模型样本中标签全为1或者0时会报错，样本不均衡导致\n')
            pass
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'models/pnn.pkl')
    torch.save(model.state_dict(), 'models/optimizer.pkl')
    # print('epoch:{} loss:{:.4f}'.format(epoch, np.mean(loss_list)))
    history.loc[epoch, ['epoch', 'loss', metric_name]] = epoch, np.mean(loss_list), np.mean(
        metric_list)


def eval(epoch):
    model = PNN(mode='out')
    model.load_state_dict(torch.load('models/5.PNN.pkl'),strict=False)
    val_loader = get_dataloader(mode='val')
    loss_function = nn.BCELoss()
    val_loss = []
    val_metric = []
    for idx, (inputs, labels) in enumerate(val_loader):
        with torch.no_grad():
            predictions = model(inputs)
            loss = loss_function(predictions, labels.float())
            val_loss.append(loss.item())
            try:
                val_metric.append(metric_func(predictions, labels).item())
            except ValueError:
                # print('二元分类模型样本中标签全为1或者0时会报错，样本不均衡导致\n')
                pass
    history.loc[epoch, ['val_loss', ('val_' + metric_name)]] = np.mean(val_loss), np.mean(val_metric)


def plot_metric(metric):
    train_metrics = history[metric]
    val_metrics = history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo-')  # 蓝色圆点实线，b——blue，o——circle marker
    plt.plot(epochs, val_metrics, 'ro--')  # 红色圆点虚线
    plt.title('PNN Training and validation_' + metric)
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(['train_' + metric, 'val_' + metric])
    plt.show()


if __name__ == '__main__':
    for i in range(5):
        train(i)
        eval(i)
    plot_metric('loss')
    plot_metric(metric_name)
    history.to_csv('training_tmp/history.csv',index=None)
