#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.datasets as datasets
import argparse
from utils import *
from utils.cand_datasets import CandDataset

DATADIR = './data/'
CKPTDIR = './annotators/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Generate annotated dataset')
parser.add_argument('--dset_name', '-ds', type=str, default='mnist', choices=['mnist', 'fashion', 'kuzushiji', 'cifar10'])
parser.add_argument('--num_classes', '-K', type=int, default=10)
parser.add_argument('--num_candidates', '-N', type=int, default=9)
parser.add_argument('--train_batch_size', '-train_bs', type=int, default=64)
parser.add_argument('--test_batch_size', '-test_bs', type=int, default=64)
parser.add_argument('--num_epochs', '-ep', type=int, default=200)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.05)
parser.add_argument('--momentum', '-mt', type=float, default=0.9)
parser.add_argument('--weight_decay', '-wd', type=float, default=5e-4)
parser.add_argument('--gamma', '-ga', type=float, default=0.5)
parser.add_argument('--step_size', '-ss', type=float, default=30)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()


def train():
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(ord_train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss/(batch_idx+1)


def test():
    model.eval()
    test_loss, correct, count = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            count += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100*correct/count


def predict(predict_loader):
    model.eval()
    generation_prob = torch.FloatTensor()
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(predict_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).cpu()
            generation_prob = torch.cat([generation_prob, outputs.data], dim=0)
    return generation_prob


def train_annotator(best_acc):
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train()
        test_loss, test_acc = test()
        print('epoch %d/%d: train_loss: %.4f test_loss: %.4f test_acc: %.4f' %\
            (epoch+1, args.num_epochs, train_loss, test_loss, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, ckptpath, start_epoch=epoch+1, best_acc=best_acc)

        scheduler.step()

    return best_acc


def annotate_dataset():
    print("Number of candidate labels:",str(args.num_candidates), '/', str(args.num_classes))
    print("Accuracy of annotator: ", best_acc)
    print("==> Calculateing generation probabilities for training set")
    generation_prob = predict(cand_train_loader)
    CandDataset(DATADIR, args.dset_name, args.num_classes, args.num_candidates, train=True, annotate=True, \
        original_dataset=cand_train_set, generation_prob=generation_prob)

    print("==> Calculateing generation probabilities for test set\n")
    generation_prob = predict(test_loader)
    CandDataset(DATADIR, args.dset_name, args.num_classes, args.num_candidates, train=False, annotate=True, \
        original_dataset=test_set, generation_prob=generation_prob)


if __name__ == '__main__':
    if args.dset_name == 'mnist':
        dataloader = datasets.MNIST
    elif args.dset_name == 'fashion':
        dataloader = datasets.FashionMNIST
    elif args.dset_name == 'kuzushiji':
        dataloader = datasets.KMNIST
    elif args.dset_name == 'cifar10':
        dataloader = datasets.CIFAR10

    assert args.num_candidates >= 1 or args.num_classes > args.num_candidates, 'Invalid number of candidates.'

    filename = 'K' + str(args.num_classes) + '_annotator.ckpt'

    ckptpath, exist = get_available_filepath(CKPTDIR, args.dset_name, filename)

    ord_train_loader, cand_train_loader, test_loader\
        = get_dataloaders_for_annotation(dataloader, DATADIR, args.num_classes, args.train_batch_size, args.test_batch_size)
    cand_train_set, test_set = cand_train_loader.dataset, test_loader.dataset

    model = CIFARAnnotator(args.num_classes) if args.dset_name == 'cifar10' else MNISTAnnotator(args.num_classes)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    if exist or args.resume:
        checkpoint = load_checkpoint(ckptpath)
        model.load_state_dict(checkpoint['model'])
        start_epoch, best_acc = checkpoint['start_epoch'], checkpoint['best_acc']
    else:
        start_epoch, best_acc = 0, 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if not exist or args.resume:
        best_acc = train_annotator(best_acc)

    annotate_dataset()


