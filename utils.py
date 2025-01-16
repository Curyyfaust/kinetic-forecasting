# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:54:52 2025

@author: lenovo
"""

import torch
import torch.nn as nn

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        """
        :param quantiles: 分位数列表，例如 [0.1, 0.5, 0.9]
        """
        super(MultiQuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, predictions, targets):
        """
        计算多维分位数损失
        :param predictions: 预测值，形状为 (batch_size, num_targets, num_quantiles)
        :param targets: 真实值，形状为 (batch_size, num_targets, 1)
        :return: 多维分位数损失
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # 计算误差，形状为 (batch_size, num_targets, 1)
            errors = targets - predictions[:, :, i:i+1]
            # 分位数损失，形状为 (batch_size, num_targets, 1)
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss)

        # 将所有分位数的损失相加，形状为 (batch_size, num_targets, 1)
        total_loss = torch.sum(torch.stack(losses, dim=-1), dim=-1)
        
        # 对所有样本和目标变量取平均
        return torch.mean(total_loss)
    



