from __future__ import print_function # Import Print function

import os

import torch
import json
import numpy as np

from woollylib.utils.utils import get_device
from woollylib.dataset import get_cifar_loader, get_advance_cifar_loader # Load test and train loaders
from woollylib.utils.transform import BASE_PROFILE, get_transform # Get transforme functions
from woollylib.utils.visualize import print_class_scale, print_samples, print_samples_native
from woollylib.models import resnet
from woollylib.backpropagation import train, train_ricap, train_native, test, test_native, get_sgd_optimizer#, get_adam_optimizer, get_crossentropy_criteria
from woollylib.utils.utils import initialize_weights, print_modal_summary, print_summary
from woollylib.scheduler import one_cycle_lr_pt, one_cycle_lr_custom
from woollylib.training import Training
from woollylib.utils.gradcam.compute import compute_gradcam

from woollylib.utils.visualize import plot_network_performance
from woollylib.utils.utils import get_incorrrect_predictions
from woollylib.utils.visualize import plot_incorrect_predictions

from woollylib.utils.utils import get_all_predictions, get_incorrrect_predictions, prepare_confusion_matrix
from woollylib.utils.visualize import plot_confusion_matrix



torch.manual_seed(1)


### Enclos inside a class

batch_size = 64
ricap_beta = 0.4
class_map = {
    'PLANE': 0,
    'CAR': 1,
    'BIRD': 2,
    'CAT': 3,
    'DEER': 4,
    'DOG': 5,
    'FROG': 6,
    'HORSE': 7,
    'SHIP': 8,
    'TRUCK': 9
}

# Enable or disable visualizations
show_summary = True
show_dataset_analyze = True


train_profile = {
    'normalize': BASE_PROFILE['normalize'],
    'shift_scale_rotate': BASE_PROFILE['shift_scale_rotate'],
    'pad_and_crop': BASE_PROFILE['pad_and_crop'],
#     'crop_and_pad': BASE_PROFILE['crop_and_pad'],
    'random_brightness_contrast': BASE_PROFILE['random_brightness_contrast'],
    'horizontal_flip': BASE_PROFILE['horizontal_flip'],
    'to_gray': BASE_PROFILE['to_gray'],
    'coarse_dropout': BASE_PROFILE['coarse_dropout'],
    'to_tensor':  BASE_PROFILE['to_tensor'],
}

train_profile['pad_and_crop']['pad'] = 4
train_profile['pad_and_crop']['p'] = 0.5

train_profile['coarse_dropout']['min_height'] = 16
train_profile['coarse_dropout']['min_width'] = 16

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
use_cuda, device = get_device()

#train_loader, test_loader = get_cifar_loader(get_transform(train_profile), get_transform(normalize), batch_size=batch_size, use_cuda=use_cuda)


