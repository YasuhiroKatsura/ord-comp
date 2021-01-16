#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from operator import mul
from functools import reduce
import itertools
import os
import random


class CandDataset(Dataset):
    training_file = '_training.pt'
    test_file = '_test.pt'

    def __init__(self, root, dset_name, num_classes, num_candidates, train=True, annotate = False, \
        original_dataset=None, generation_prob=None, num_data=5000, transform=None, candidates_transform=None):

        self.processed_dir = os.path.join(root, dset_name + '-annotated')
        self.train = train
        self.transform = transform
        self.candidates_transform = candidates_transform
        self.num_classes = num_classes
        self.num_candidates = num_candidates
        self.class_list = list(range(self.num_classes))
        self.num_data = num_data

        filename = 'K' + str(self.num_classes)
        if self.train:
            filename += ('_N' + str(self.num_candidates) + self.training_file)
        else:
            filename += self.test_file

        datapath = os.path.join(self.processed_dir, filename)

        if annotate:
            self.original_targets = original_dataset.targets

            if not os.path.exists(datapath):
                data, candidates = self._annotate_data(original_dataset.data, generation_prob)

                if not os.path.exists(self.processed_dir):
                    os.makedirs(self.processed_dir)
                print("Saving cand dataset")
                torch.save((data, candidates, self.original_targets), datapath)

                self.data, self.candidates = data, candidates

        else:
            self.data, self.candidates, self.original_targets = torch.load(datapath)

            if self.train:
                print("dataset name: %s\nnumber of classes: %d\nnumber of candidates: %d\n--" %\
                    (dset_name, self.num_classes, self.num_candidates))
                self.data, self.candidates = self._filter_excess_data()
            
            else:
                print("cand dataset for test")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img, candidates = self.data[index], self.candidates[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.candidates_transform is not None:
            candidates = self.candidates_transform(candidates)

        return img, candidates


    def _filter_excess_data(self):

        count_list, delete_index_list = [0 for i in self.class_list], []
        indices_list = list(range(len(self.data)))
        random.shuffle(indices_list)

        if self.train:
            for index in indices_list:
                target = self.original_targets[index]

                if count_list[target] >= self.num_data:
                    delete_index_list.append(index)
                else:
                    count_list[target] += 1

            data = np.delete(self.data, delete_index_list, 0)
            candidates = np.delete(self.candidates, delete_index_list, 0)

        return data, candidates


    def _annotate_data(self, data, generation_prob):
        print("==> Annotating Dataset")
        data = data.numpy() if type(data) is torch.Tensor else data
        generation_prob = generation_prob.numpy() if type(generation_prob) is torch.Tensor else generation_prob

        num_candidates = self.num_candidates if self.train else 1

        normalization_const = self._get_normalization_const()
        candidates = np.empty([0, num_candidates], dtype=int)
        candidate_comb_list = [i for i in itertools.combinations(self.class_list, num_candidates)]
        candidate_comb_list_indices = np.arange(len(candidate_comb_list))

        for row in generation_prob:
            prob_comb_list = np.array([sum([row[c] for c in comb]) for comb in candidate_comb_list]) * normalization_const
            prob_comb_list /= prob_comb_list.sum()
            candidate = candidate_comb_list[np.array(np.random.choice(candidate_comb_list_indices, p=prob_comb_list))]
            candidates = np.r_[candidates, np.array(candidate).reshape(-1, num_candidates)]

        return data, candidates


    def _get_normalization_const(self):
        n, r = self.num_classes - 1, self.num_candidates - 1
        r = min(n - r, r)
        if r == 0: return 1
        over = reduce(mul, range(n, n - r, -1))
        under = reduce(mul, range(1,r + 1))
        return under / over
