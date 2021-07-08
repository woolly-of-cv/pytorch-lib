# Libs
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # The GPU id to use, "0" to  "7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.join(os.getcwd(), "pytorch-lib/"))


import time
import torch
from torch import nn
import torchvision

import numpy as np

from collections import namedtuple
import copy

import functools
from functools import lru_cache as cache
from functools import partial

import matplotlib.pyplot as plt

from woollylib.utils.utils import get_device

(use_cuda, device) = get_device()
cpu = torch.device('cpu')

#####################
# Augmentations
#####################


class Crop(namedtuple('Crop', ('h', 'w'))):
    def apply(self, x, x0, y0):
        return x[..., y0:y0+self.h, x0:x0+self.w]

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]


class FlipLR(namedtuple('FlipLR', ())):
    def apply(self, x, choice):
        return flip_lr(x) if choice else x

    def options(self, shape):
        return [{'choice': b} for b in [True, False]]


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def apply(self, x, x0, y0):
        x[..., y0:y0+self.h, x0:x0+self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]

# Load CIFAR and preprocess it


def chunks(data, splits): return (data[start:end]
                                  for (start, end) in zip(splits, splits[1:]))


def even_splits(N, num_chunks): return np.cumsum(
    [0] + [(N//num_chunks)+1]*(N % num_chunks) + [N//num_chunks]*(num_chunks - (N % num_chunks)))


def shuffled(xs, inplace=False):
    xs = xs if inplace else copy.copy(xs)
    np.random.shuffle(xs)
    return xs


def transformed(data, targets, transform, max_options=None, unshuffle=False):
    i = torch.randperm(len(data), device=device)
    data = data[i]
    options = shuffled(transform.options(data.shape),
                       inplace=True)[:max_options]
    data = torch.cat([transform.apply(x, **choice) for choice,
                     x in zip(options, chunks(data, even_splits(len(data), len(options))))])
    return (data[torch.argsort(i)], targets) if unshuffle else (data, targets[i])


class Batches():
    def __init__(self, batch_size, transforms=(), dataset=None, shuffle=True, drop_last=False, max_options=None):
        self.dataset, self.transforms, self.shuffle, self.max_options = dataset, transforms, shuffle, max_options
        N = len(dataset['data'])
        self.splits = list(range(0, N+1, batch_size))
        if not drop_last and self.splits[-1] != N:
            self.splits.append(N)

    def __iter__(self):
        data, targets = self.dataset['data'], self.dataset['targets']
        for transform in self.transforms:
            data, targets = transformed(
                data, targets, transform, max_options=self.max_options, unshuffle=not self.shuffle)
        if self.shuffle:
            i = torch.randperm(len(data), device=device)
            data, targets = data[i], targets[i]
        return ((x.clone(), y) for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits)))

    def __len__(self):
        return len(self.splits) - 1


@cache(None)
def cifar10(root='./data'):
    def download(train): return torchvision.datasets.CIFAR10(
        root=root, train=train, download=True)
    return {k: {'data': torch.tensor(v.data), 'targets': torch.tensor(v.targets)}
            for k, v in [('train', download(True)), ('valid', download(False))]}


cifar10_mean, cifar10_std = [
    # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
    (125.31, 122.95, 113.87),
    # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
    (62.99, 62.09, 66.70),
]
cifar10_classes = 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(
    ', ')


class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t


mean, std = [torch.tensor(x, device=device, dtype=torch.float16)
             for x in (cifar10_mean, cifar10_std)]


def normalise(data, mean=mean, std=std):
    return (data - mean)/std


def unnormalise(data, mean=mean, std=std):
    return data*std + mean


def pad(data, border):
    return nn.ReflectionPad2d(border)(data)


transpose = lambda x, source='NHWC', target='NCHW': x.permute(
    [source.index(d) for d in target])
to = lambda *args, **kwargs: (lambda x: x.to(*args, **kwargs))


def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms):
        dataset['data'] = transform(dataset['data'])
    return dataset


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def flip_lr(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, [-1])
    return x[..., ::-1].copy()


def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k, v in nested_dict.items()}


def image_plot(ax, img, title):
    ax.imshow(to_numpy(unnormalise(transpose(img, 'CHW', 'HWC'))).astype(np.int))
    ax.set_title(title)
    ax.axis('off')


def layout(figures, sharex=False, sharey=False, figure_title=None, col_width=4, row_height=3.25, **kw):
    nrows, ncols = np.array(figures).shape

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex,
                            sharey=sharey, figsize=(col_width*ncols, row_height*nrows))
    axs = [figure(ax, **kw) for row in zip(np.array(axs).reshape(nrows,
                                                                 ncols), figures) for ax, figure in zip(*row)]
    fig.suptitle(figure_title)
    return fig, axs


def train_test_load(batch_size, use_cuda, ricap_beta):
    dataset = cifar10()  # downloads dataset
    dataset = map_nested(to(device), dataset)
    train_set = preprocess(dataset['train'], [partial(
        pad, border=4), transpose, normalise, to(torch.float32)])
    valid_set = preprocess(
        dataset['valid'], [transpose, normalise, to(torch.float32)])

    train_batches = partial(Batches, dataset=train_set,
                            shuffle=True,  drop_last=False, max_options=200)
    valid_batches = partial(Batches, dataset=valid_set,
                            shuffle=False, drop_last=False)


    return train_batches(batch_size=batch_size, transforms=(Crop(32, 32), FlipLR(), Cutout(8, 8))), valid_batches(batch_size=batch_size)