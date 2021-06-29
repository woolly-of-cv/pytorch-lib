from albumentations.augmentations.transforms import Cutout, HorizontalFlip
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

from torchvision import transforms

torch.manual_seed(1)


BASE_PROFILE = {
    'shift_scale_rotate': {
        'shift_limit': 0.15,
        'scale_limit': 0.15,
        'rotate_limit': 10,
        'p': 0.3
    },
    'random_resized_crop': {
        'height': 32,
        'width': 32,
        'scale': (0.8, 1.0),
        'p': 0.5
    },
    'crop_and_pad': {
        'px': (0, 6),
        'pad_mode': cv.BORDER_REPLICATE,
        'p': 0.3
    },
    'random_brightness_contrast': {
        'p': 0.3
    },
    'gauss_noise': {
        'p': 0.2
    },
    'equalize': {
        'p': 0.2
    },
    'horizontal_flip': {
        'p': 0.3
    },
    'to_gray': {
        'p': 0.3
    },
    'coarse_dropout': {
        'max_holes': 1,
        'max_height': 16,
        'max_width': 16,
        'min_holes': 1,
        'min_height': 8,
        'min_width': 8,
        'fill_value': (0.4914, 0.4822, 0.4465),
        'p': 0.3
    },
    'normalize': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616)
    }
}


def get_transform(profile):
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """

    trs = []
    if 'shift_scale_rotate' in profile:
        ssr = profile['shift_scale_rotate']
        trs.append(
            A.ShiftScaleRotate(
                shift_limit=ssr['shift_limit'], 
                scale_limit=ssr['scale_limit'], 
                rotate_limit=ssr['rotate_limit'], 
                p=ssr['p']
            )
        )

    if 'random_resized_crop' in profile:
        rrp = profile['random_resized_crop']
        trs.append(
            A.RandomResizedCrop(
                height=rrp['height'], 
                width=rrp['width'], 
                scale=rrp['scale'], 
                p=rrp['p']
            )
        )

    if 'crop_and_pad' in profile:
        cap = profile['crop_and_pad']
        trs.append(
            A.CropAndPad(
                px=cap['px'],
                pad_mode=cap['pad_mode'],
                p=cap['p'], 
            )
        )

    if 'random_brightness_contrast' in profile:
        rbc = profile['random_brightness_contrast']
        trs.append(
            A.RandomBrightnessContrast(
                p=rbc['p'], 
            )
        )

    if 'gauss_noise' in profile:
        gn = profile['gauss_noise']
        trs.append(
            A.GaussNoise(
                p=gn['p'], 
            )
        )

    if 'equalize' in profile:
        eq = profile['equalize']
        trs.append(
            A.Equalize(
                p=eq['p'], 
            )
        )

    if 'horizontal_flip' in profile:
        hf = profile['horizontal_flip']
        trs.append(
            A.HorizontalFlip(
                p=hf['p'], 
            )
        )

    if 'to_gray' in profile:
        tg = profile['to_gray']
        trs.append(
            A.ToGray(
                p=tg['p']
            )
        )
    
    if 'normalize' in profile:
        norm = profile['normalize']
        trs.append(
            A.Normalize(
                mean=norm['mean'],
                std=norm['std'], 
            )
        )

    if 'coarse_dropout' in profile:
        cd = profile['coarse_dropout']
        trs.append(
            A.CoarseDropout(
                max_holes=cd['max_holes'], 
                max_height=cd['max_height'], 
                max_width=cd['max_width'], 
                min_holes=cd['min_holes'],
                min_height=cd['min_height'], 
                min_width=cd['min_width'], 
                fill_value=cd['fill_value'], 
                p=cd['p']
            )
        )

    trs.append(ToTensorV2())

    return A.Compose(trs)


def get_p_train_transform():
    """Get Pytorch Transform function for train data

    Returns:
        Compose: Composed transformations
    """
    random_rotation_degree = 5
    img_size = (28, 28)
    random_crop_percent = (0.85, 1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, random_crop_percent),
        transforms.RandomRotation(random_rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])


def get_p_test_transform():
    """Get Pytorch Transform function for test data

    Returns:
        Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
