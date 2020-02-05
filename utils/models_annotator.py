#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as cnn
import math

torch.manual_seed(0)


class MNISTAnnotator(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTAnnotator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CIFARAnnotator(nn.Module):
    def __init__(self, num_classes):
        super(CIFARAnnotator, self).__init__()
        self.base_model = cnn.resnext101_32x8d(num_classes=num_classes)

    def forward(self, x):
        return F.softmax(self.base_model(x), dim=1)


