#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class CandLoss(nn.Module):

    def __init__(self, binary_loss, device):
        super(CandLoss, self).__init__()
        self.device = device
        if binary_loss == 'sigmoid': self.binary_loss = lambda x: torch.sigmoid(-x) -1/2
        elif binary_loss == 'ramp': self.binary_loss = lambda x: F.hardtanh(-x) -1/2


    def forward(self, outputs, candidates):
        loss=torch.tensor([0.0]).to(self.device)

        for i in range(candidates.shape[1]):
            loss += self.multi_loss(outputs, candidates[:, i:i+1])
        return loss/len(outputs)


class OVALoss(CandLoss):

    def __init__(self, binary_loss, device):
        super(OVALoss, self).__init__(binary_loss, device)


    def multi_loss(self, outputs, candidates):
        term1 = self.binary_loss(outputs.gather(1, candidates))
        term2 = torch.sum(self.binary_loss(-outputs), dim=1).view(-1, 1) - self.binary_loss(-outputs.gather(1, candidates))
        return torch.sum(term1 + term2/(len(outputs[0])-1))


class PCLoss(CandLoss):

    def __init__(self, binary_loss, device):
        super(PCLoss, self).__init__(binary_loss, device)


    def multi_loss(self, outputs, candidates):
        candidates = outputs.gather(1, candidates)
        candidates = torch.stack([candidates for i in range(outputs.shape[1])], dim=1).view_as(outputs)
        zeros = torch.zeros(outputs.shape[0], 1).to(self.device)

        term1 = torch.sum(self.binary_loss(candidates-outputs), dim=1)
        term2 = torch.sum(self.binary_loss(zeros), dim=1)

        return torch.sum(term1 - term2)


