#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as cnn
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.conv2(F.relu(self.conv1(x))))
        x = x.view(-1, 64*12*12)
        out = self.fc2(F.relu(self.fc1(x)))
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=500):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.squeeze(x.view(-1, self.input_dim), 0)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


class Linear(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.squeeze(x.view(-1, self.input_dim), 0)
        x = torch.squeeze(x, 0)
        out = self.linear(x)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.base_model = cnn.densenet161(num_classes=num_classes)

    def forward(self, x):
        return F.softmax(self.base_model(x), dim=1)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.base_model = cnn.resnet152(num_classes=num_classes)

    def forward(self, x):
        return F.softmax(self.base_model(x), dim=1)


