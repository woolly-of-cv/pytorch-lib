import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from torch.nn.modules.loss import _WeightedLoss

torch.manual_seed(1)


def train(use_l1=False, lambda_l1=5e-4):
    """ Function to return train function instance

    Args:
        use_l1 (bool, optional): Enable L1. Defaults to False.
        lambda_l1 (float, optional): L1 Value. Defaults to 5e-4.
    """
    def internal(model, train_loader, optimizer, criteria, dropout, device, scaler=None, scheduler=None):
        """ This function is for running backpropagation

        Args:
            model (Net): Model instance to train
            train_loader (Dataset): Dataset used in training
            optimizer (torch.optim): Optimizer used
            dropout (bool): Enable/Disable 
            device (string, cuda/cpu): Device type Values Allowed - cuda/cpu
            scheduler (Scheduler, optional): scheduler instance used for updating lr while training. Defaults to None.

        Returns:
            (float, int): Loss, Number of correct Predictions
        """
        model.train()
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, dropout)
            loss = criteria(output, target)
            if use_l1 == True:
                l1 = 0
                for p in model.parameters():
                    l1 = l1 + p.square().sum()
                loss = loss + lambda_l1 * l1
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader), 100. * correct / len(train_loader.dataset)

    return internal


def test(model, test_loader, criteria, device):
    """ Function to perform model validation

    Args:
        model (Net): Model instance to run validation
        test_loader (Dataset): Dataset used in validation
        device (string, cuda/cpu): Device type Values Allowed - cuda/cpu

    Returns:
        (float, int): Loss, Number of correct Predictions
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criteria(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss/len(test_loader.dataset), 100. * correct / len(test_loader.dataset)