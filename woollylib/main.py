from __future__ import print_function # Import Print function

import os

import torch
import json
import numpy as np

from woollylib.dataset import get_cifar_loader, get_advance_cifar_loader # Load test and train loaders
from woollylib.utils.transform import BASE_PROFILE, get_transform # Get transforme functions
from woollylib.utils.visualize import print_samples
from woollylib.bp.optimizer.backpropagation import get_sgd_optimizer
from woollylib.bp.losses.backpropagation import  get_crossentropy_criteria, get_label_smoothing_criteria
from woollylib.scheduler import one_cycle_lr_custom

from woollylib.utils.gradcam.compute import compute_gradcam

torch.manual_seed(1)
global train_profile

def train_test_load(batch_size, use_cuda, ricap_beta):
    train_profile = {
        'normalize': BASE_PROFILE['normalize'],
        'shift_scale_rotate': BASE_PROFILE['shift_scale_rotate'],
#         'rotate': BASE_PROFILE['rotate'],
        'pad_and_crop': BASE_PROFILE['pad_and_crop'],
#         'crop_and_pad': BASE_PROFILE['crop_and_pad'],
#         'random_brightness_contrast': BASE_PROFILE['random_brightness_contrast'],
        'horizontal_flip': BASE_PROFILE['horizontal_flip'],
#         'to_gray': BASE_PROFILE['to_gray'],
        'coarse_dropout': BASE_PROFILE['coarse_dropout'],
        'to_tensor':  BASE_PROFILE['to_tensor'],
    }

    train_profile['shift_scale_rotate']['rotate_limit'] = 5
    train_profile['shift_scale_rotate']['p'] = 0.5
    
    train_profile['pad_and_crop']['pad'] = 4
    train_profile['pad_and_crop']['p'] = 1.0
    
#     train_profile['rotate']['limit'] = 5
#     train_profile['rotate']['p'] = 1.0

    train_profile['coarse_dropout']['min_height'] = 8
    train_profile['coarse_dropout']['min_width'] = 8
    train_profile['coarse_dropout']['p'] = 1.0
    train_profile['coarse_dropout']['max_height'] = 8
    train_profile['coarse_dropout']['max_width'] = 8
#     train_profile['coarse_dropout']['fill_value'] = 0

    normalize = {
        'normalize': BASE_PROFILE['normalize'],
        'to_tensor':  BASE_PROFILE['to_tensor'],
    }

    course_dropout = {
        'coarse_dropout': BASE_PROFILE['coarse_dropout'],
    }

    ricap_profile = {
        'p': 0.3,
        'ricap_beta': ricap_beta
    }

    train_loader, test_loader = get_cifar_loader(get_transform(train_profile), get_transform(normalize), batch_size=batch_size, use_cuda=use_cuda)
    return train_loader, test_loader

def get_optimizer(model,lr, momentum, weight_decay, device):
    optimizer = get_sgd_optimizer(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criteria = get_label_smoothing_criteria(device)
    return optimizer, criteria

def get_scheduler(epochs, lr, max_lr, mom, mom_max, optimizer, steps_per_epoch):
    schedule = np.interp(np.arange(epochs+1), [0, 4, epochs], [lr, max_lr, lr/20.0])
    mschedule = np.interp(np.arange(epochs+1), [0, 4, epochs], [mom_max, mom, mom_max])
    # Create Custom One Cycle schedule instance
    custom_scheduler = one_cycle_lr_custom(
        optimizer, 
        lr=lr, 
        max_lr=max_lr, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs,
        lrschedule=schedule,
        mschedule=mschedule
    )
    return custom_scheduler

def get_samples_visualize(show_dataset_analyze, train_loader, class_map):
    if show_dataset_analyze:
        print_samples(train_loader, class_map)