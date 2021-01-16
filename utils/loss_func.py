#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class OVALoss():
    def __init__(self, binary_loss):
        self.binary_loss = binary_loss


    def loss(self, outputs, targets):
        # outputs = F.softmax(outputs, dim=1) - 1/2 # leave this out if not avaluate error with Rademacher complexity = 1/2

        term1 = self.binary_loss(outputs.gather(1, targets))
        term2 = torch.sum(self.binary_loss(-outputs), dim=1).view(-1, 1) - self.binary_loss(-outputs.gather(1, targets))
        return torch.sum(term1 + term2/(len(outputs[0])-1))


class PCLoss():

    def __init__(self, device, binary_loss):
        self.device = device
        self.binary_loss = binary_loss


    def loss(self, outputs, targets):
        # outputs = F.softmax(outputs, dim=1) - 1/2 # leave this out if not avaluate error with Rademacher complexity = 1/2

        targets = outputs.gather(1, targets)
        targets = torch.stack([targets for i in range(outputs.shape[1])], dim=1).view_as(outputs)
        zeros = torch.zeros(outputs.shape[0], 1).to(self.device)

        term1 = torch.sum(self.binary_loss(targets-outputs), dim=1)
        term2 = torch.sum(self.binary_loss(zeros), dim=1)

        return torch.sum(term1 - term2)


class CELoss():
    
    def __init__(self):
        pass


    def loss(self, outputs, targets):
        loss = torch.nn.CrossEntropyLoss()
        return loss(outputs, targets.view(outputs.shape[0]))


class CandLoss(nn.Module):

    def __init__(self, device, multi_loss, binary_loss=None, unbiased=False, num_classes=10, num_candidates=9):
        super(CandLoss, self).__init__()
        self.device = device
        self.unbiased = unbiased
        self.num_classes = num_classes
        self.num_candidates = num_candidates
        self.class_list = list(range(self.num_classes))

        if binary_loss == 'sigmoid': binary_loss = lambda x: torch.sigmoid(-x) -1/2
        elif binary_loss == 'ramp': binary_loss = lambda x: F.hardtanh(-x) -1/2

        self.multi_loss = multi_loss
        if self.multi_loss == 'ova':
            self.loss = OVALoss(binary_loss).loss
        elif self.multi_loss == 'pc':
            self.loss = PCLoss(device, binary_loss).loss
        elif self.multi_loss == 'ce':
            self.loss = CELoss().loss


    def forward(self, outputs, candidates):
        loss=torch.tensor([0.0]).to(self.device)

        if self.multi_loss=='ova' or self.multi_loss=='pc' or not self.unbiased:
            for i in range(candidates.shape[1]):
                loss += self.loss(outputs, candidates[:, i:i+1])
        else:
            for i in range(candidates.shape[1]):
                loss += (self.num_classes - 1) * self.loss(outputs, candidates[:, i:i+1])
            loss -= (self.num_candidates - 1) * self.total_loss(outputs)         

        return loss/len(outputs)


    def total_loss(self, outputs):
        loss=torch.tensor([0.0]).to(self.device)
        all_classes = torch.tensor([self.class_list for i in range(len(outputs))]).to(self.device)

        for i in range(self.num_classes):
            loss += self.loss(outputs, all_classes[:, i:i+1].view(outputs.shape[0]))
        return loss
