import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import os

from woollylib.utils.transform import convert_to_tensor

torch.manual_seed(1)

class WyCustomDataset(Dataset):
    def __init__(self, class_map, path='./data', transforms=None):
        self.img_dim = (32, 32)
        self.transforms = transforms
        self.idata = []
        self.ilabel = []
        
        self.class_map = class_map
        
        # Read Dataset
        self.images = glob.glob(path + '/**/*.jpg')
        
        for image in self.images:
            self.ilabel.append(self.class_map[image.split('/')[2].upper()])
            self.idata.append(image)

    def __len__(self):
        return len(self.idata)

    def __getitem__(self, idx):
        # Read Image and Label
        im_path = self.idata[idx]
        label = self.ilabel[idx]
        
        # open method used to open different extension image file
        with Image.open(im_path) as f:
            image: np.ndarray = np.asarray(f) #.convert('RGB'))
        
        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
            
        return image, label, im_path

class WyDatasetAdvance(Dataset):
    """
    Custom Dataset Class

    """

    def __init__(self, dataset, transforms=None, normalize=None, apply_ricap=None, apply_cutout=None):
        """Initialize Dataset

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        self.transforms = transforms
        self.normalize = normalize
        self.apply_ricap = apply_ricap
        self.apply_cutout = apply_cutout
        self.dataset = dataset
        self.image_width, self.image_height = self.dataset[0][0].size

    def __len__(self):
        """Get dataset length

        Returns:
            int: Length of dataset
        """
        return len(self.dataset)

    def ricap(self, ricap_beta=0.3):
        # I_x, I_y = data.size()[2:]
        image_width, image_height = self.image_width, self.image_height

        # Find random height and width for images
        random_height = int(
            np.round(image_height * np.random.beta(ricap_beta, ricap_beta)))
        random_width = int(
            np.round(image_width * np.random.beta(ricap_beta, ricap_beta)))

        # Calculate each block height and width
        cropped_width = [random_width, image_width -
                         random_width, random_width, image_width - random_width]
        cropped_height = [random_height, random_height,
                          image_height - random_height, image_height - random_height]

        cropped_images = {}

        labels = [0]*4
        weightage = [0.0]*4

        for k in range(4):
            idx = torch.randint(low=0, high=len(self.dataset), size=(1,))
            start_y = np.random.randint(
                0, image_height - cropped_height[k] + 1)
            start_x = np.random.randint(0, image_width - cropped_width[k] + 1)
            image = np.array(self.dataset[idx][0])
            # Apply Transforms
            image = self.apply_transform(image)
            cropped_images[k] = image[start_x:start_x +
                                      cropped_width[k], start_y:start_y + cropped_height[k], :]
            labels[k] = self.dataset[idx][1]
            weightage[k] = cropped_width[k] * \
                cropped_height[k] / (image_width * image_height)

        patched_image = np.concatenate(
            (np.concatenate((cropped_images[0], cropped_images[1]), 0),
             np.concatenate((cropped_images[2], cropped_images[3]), 0)),
            1
        )

        return patched_image, labels, weightage, True

    def apply_transform(self, image):
        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.normalize is not None:
            image = self.normalize(image=image)["image"]

        return image

    def __getitem__(self, idx):
        """Get an item from dataset

        Args:
            idx (int): id of item in dataset

        Returns:
            (tensor, int): Return tensor of transformer image, label
        """

        # Read Image and Label
        image, label = self.dataset[idx]

        label = [label]*4
        weightage = [1.0, 0.0, 0.0, 0.0]
        image = np.array(image)
        applied_ricap = False

        # Apply Transforms
        image = self.apply_transform(image)

        if self.apply_ricap and self.apply_cutout:
            recap_prob = self.apply_ricap['p']
            ricap_beta = self.apply_ricap['ricap_beta']

            if np.random.rand() < recap_prob:
                image, label, weightage, applied_ricap = self.ricap(ricap_beta=ricap_beta)
            else:
                image = self.apply_cutout(image=image)["image"]
        elif self.apply_ricap:
            recap_prob = self.apply_ricap['p']
            ricap_beta = self.apply_ricap['ricap_beta']

            if np.random.rand() < recap_prob:
                image, label, weightage, applied_ricap = self.ricap(ricap_beta=ricap_beta)
        elif self.apply_cutout:
            image = self.apply_cutout(image=image)["image"]

        image = convert_to_tensor(image=image)["image"]

        return image, label, weightage, applied_ricap


class WyDataset(Dataset):
    """
    Custom Dataset Class

    """

    def __init__(self, dataset, transforms=None):
        """Initialize Dataset

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        """Get dataset length

        Returns:
            int: Length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get an item form dataset

        Args:
            idx (int): id of item in dataset

        Returns:
            (tensor, int): Return tensor of transformer image, label
        """
        # Read Image and Label
        image, label = self.dataset[idx]

        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return (image, label)


def get_mnist_loader(train_transform, test_transform, batch_size=64, use_cuda=True):
    """Get instance of train and test loaders for MNIST data

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to True.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        WyDataset(datasets.MNIST('../data', train=True,
                                 download=True), transforms=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        WyDataset(datasets.MNIST('../data', train=False,
                                 download=True), transforms=test_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def get_cifar_loader(train_transform, test_transform, batch_size=64, use_cuda=True):
    """Get instance of train and test loaders for CIFAR10 data

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to True.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        WyDataset(datasets.CIFAR10('./data', train=True,
                                   download=True), transforms=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        WyDataset(datasets.CIFAR10('./data', train=False,
                                   download=True), transforms=test_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def get_advance_cifar_loader(
    base_transform=None,
    normalize=None,
    apply_ricap=None,
    apply_cutout=None,
    batch_size=64,
    use_cuda=True
):
    """Get instance of train and test loaders for CIFAR10 data

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to True.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        WyDatasetAdvance(
            datasets.CIFAR10(
                '../data', train=True, download=True
            ),
            transforms=base_transform,
            normalize=normalize,
            apply_ricap=apply_ricap,
            apply_cutout=apply_cutout
        ),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        WyDatasetAdvance(
            datasets.CIFAR10(
                '../data', train=False, download=True
            ),
            normalize=normalize
        ),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
