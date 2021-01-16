#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt


transform_train = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5,],[0.5,])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,],[0.5,])])


def filter_classes(dataset, num_classes):
    data, targets = dataset.data.tolist(), dataset.targets

    filter_items = lambda items,indices: [item for idx,item in enumerate(items) if idx not in indices]

    class_list = list(range(num_classes))
    remove_index_list = [i for i, target in enumerate(targets) if target not in class_list]

    data, targets = filter_items(data, remove_index_list), filter_items(targets, remove_index_list)
    dataset.data = torch.ByteTensor(data) if type(dataset.data) is torch.Tensor else np.array(data, dtype='uint8')
    dataset.targets = targets

    return dataset


def get_dataloaders_for_annotation(dataloader, datadir, num_classes, train_batch_size, test_batch_size):
    ord_train_set = filter_classes(dataloader(root=datadir, train=True, download=True), num_classes)
    cand_train_set = ord_train_set
    ord_train_set.transform, cand_train_set.transform = transform_train, transform_test

    test_set = filter_classes(dataloader(root=datadir, train=False, download=True, transform=transform_test), num_classes)

    return DataLoader(dataset=ord_train_set, batch_size=train_batch_size, shuffle=True, num_workers=2),\
           DataLoader(dataset=cand_train_set, batch_size=test_batch_size, shuffle=False, num_workers=2),\
           DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)


def get_cand_dataloader(dataloader, datadir, dset_name, num_classes, num_candidates, num_data, train_batch_size, test_batch_size):
    train_set = dataloader(root=datadir, dset_name=dset_name, num_classes=num_classes, num_candidates=num_candidates, num_data=num_data, train=True, transform=transform_train)
    test_set = dataloader(root=datadir, dset_name=dset_name, num_classes=num_classes, num_candidates=num_candidates, train=False, transform=transform_test)

    return DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=2),\
           DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)


def save_checkpoint(model, ckptpath, **kwargs):
    state = {'model': model.state_dict()}
    state.update(kwargs)
    print('==> Saving weights')
    torch.save(state, ckptpath)


def load_checkpoint(ckptpath):
    print('==> Resuming from checkpoint')
    return torch.load(ckptpath)


def imshow(img, unnormalize=True):
    if unnormalize:
        img = img / 2 + 0.5
    if type(img) is torch.Tensor:
        img = img.numpy()
    if img.shape[0]==1:
        plt.imshow(np.squeeze(img), cmap='gray_r')
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.xticks(color="None"), plt.yticks(color="None"), plt.tick_params(length=0)
    plt.show()


