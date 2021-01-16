#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.datasets as datasets
import argparse
from utils import *
from utils.cand_datasets import CandDataset as dataloader

DATADIR = './data/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Demonstrate experiment')
parser.add_argument('--dset_name', '-ds', type=str, default='mnist', choices=['mnist', 'fashion', 'kuzushiji', 'cifar10'])
parser.add_argument('--num_classes', '-K', type=int, default=10)
parser.add_argument('--num_candidates', '-N', type=int, default=9)
parser.add_argument('--num_data', '-n', type=int, default=1000)
parser.add_argument('--num_trial', '-nt', type=int, default=5)
parser.add_argument('--model', '-m', type=str, default='mlp', choices=['mlp', 'linear', 'densenet', 'resnet'])
parser.add_argument('--binary_loss', '-bl', type=str, default='sigmoid', choices=['sigmoid', 'ramp'])
parser.add_argument('--multi_loss', '-ml', type=str, default='ova', choices=['ova', 'pc', 'ce'])
parser.add_argument('--unbiased', '-ub', action='store_true')
parser.add_argument('--train_batch_size', '-train_bs', type=int, default=64)
parser.add_argument('--test_batch_size', '-test_bs', type=int, default=64)
parser.add_argument('--num_epochs', '-ne', type=int, default=300)
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5)
parser.add_argument('--momentum', '-mt', type=float, default=0.9)
parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001)
args = parser.parse_args()


def train():
    model.train()
    train_loss = 0
    num_data = 0

    for batch_idx, (inputs, candidates) in enumerate(train_loader):
        inputs, candidates = inputs.to(device), candidates.to(device)
        num_batch_data = len(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, candidates)
        train_loss += loss.item()*num_batch_data
        num_data += num_batch_data
        loss.backward()
        optimizer.step()

    return train_loss/num_data


def test():
    model.eval()
    test_loss, correct, denoise_loss, denoise_correct, num_data = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_batch_data = len(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*num_batch_data
            num_data += num_batch_data
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets.squeeze()).sum().item()

            if not args.unbiased:
                outputs = outputs/beta - (1-beta)/beta*noised_prob # denoise
                loss = criterion(outputs, targets)
                denoise_loss += loss.item()*num_batch_data
                _, predicted = outputs.max(1)
                denoise_correct += predicted.eq(targets.squeeze()).sum().item()

    if not args.unbiased:
        return test_loss/num_data, 100*correct/num_data, denoise_loss/num_data, 100*denoise_correct/num_data
    else:
        return test_loss/num_data, 100*correct/num_data


def run_demo(trial):
    if args.unbiased:
        logger = Logger(logpath + str(trial), ['epoch', 'train_loss', 'test_loss', 'acc'])
    else:
        logger = Logger(logpath + str(trial), ['epoch', 'train_loss', 'test_loss', 'acc', 'denoise_loss', 'denoise_acc'])

    print("==> Start demo")

    for epoch in range(args.num_epochs):
        train_loss = train()

        if args.unbiased:
            test_loss, test_acc = test()
            print('epoch %d/%d: train_loss: %.4f test_loss: %.4f test_acc: %.4f' %\
                  (epoch+1, args.num_epochs, train_loss, test_loss, test_acc))
            logger.add(epoch, train_loss, test_loss, test_acc)
        else:
            test_loss, test_acc, denoise_loss, denoise_acc = test()
            print('epoch %d/%d: train_loss: %.4f test_loss: %.4f test_acc: %.4f denoise_loss: %.4f denoise_acc: %.4f' %\
                  (epoch+1, args.num_epochs, train_loss, test_loss, test_acc, denoise_loss, denoise_acc))
            logger.add(epoch, train_loss, test_loss, test_acc, denoise_loss, denoise_acc)        

        if args.dset_name == 'cifar10':
            scheduler.step()

    logger.close()


if __name__ == '__main__':
    if args.dset_name == 'mnist':
        input_dim = 1*28*28
    elif args.dset_name == 'fashion':
        input_dim = 1*28*28
    elif args.dset_name == 'kuzushiji':
        input_dim = 1*28*28
    elif args.dset_name == 'cifar10':
        input_dim = 3*32*32

    start_epoch, best_acc = 0, 0
    assert args.num_candidates >= 1 or args.num_classes > args.num_candidates, 'Invalid number of candidates.'

    beta = (args.num_classes - args.num_candidates)/args.num_candidates/(args.num_classes - 1)
    noised_prob = 1/args.num_classes

    if args.multi_loss == 'ova' or args.multi_loss == 'pc':
        criterion = CandLoss(device, args.multi_loss, args.binary_loss, unbiased=True) # OVA/PC loss functions eable unbiased estimation
    elif args.multi_loss == 'ce':
        if args.unbiased:
            criterion = CandLoss(device, args.multi_loss, unbiased=True, num_classes=args.num_classes, num_candidates=args.num_candidates)
        else:
            criterion = CandLoss(device, args.multi_loss)

    logpath = get_logpath(args)

    for trial in range(args.num_trial):
        print('Number of trial:', trial, '/', args.num_trial)

        train_loader, test_loader = get_cand_dataloader(dataloader, DATADIR, args.dset_name, args.num_classes, args.num_candidates, args.num_data, args.train_batch_size, args.test_batch_size)

        if args.model == 'densenet': model = DenseNet(num_classes=args.num_classes)
        elif args.model == 'resnet': model = Resnet(num_classes=args.num_classes)
        elif args.model == 'mlp': model = MLP(input_dim, args.num_classes)
        elif args.model == 'linear': model = Linear(input_dim, args.num_classes)

        if device == 'cuda':
            model = torch.nn.DataParallel(model)

        if args.dset_name == 'cifar10':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        run_demo(trial)


