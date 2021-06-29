import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np

torch.manual_seed(1)

def train_ricap(use_l1=False, lambda_l1=5e-4, ricap_beta=0.3):
    """ Function to return train function instance

    Args:
        use_l1 (bool, optional): Enable L1. Defaults to False.
        lambda_l1 (float, optional): L1 Value. Defaults to 5e-4.
    """
    def internal(model, train_loader, optimizer, criteria, dropout, device, scheduler=None):
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

        def accuracy(output, target, topk=(1,)):
            """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                return res

        def ricap(data, target):
            eploss = 0
            acc = 0
            batch_size = data.size(0)

            # Height and Width of image
            I_x, I_y = data.size()[2:]

            # Find random height and width for images
            w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
            h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(batch_size)
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = data[idx][:, :,x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = target[idx].to(device)
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                 torch.cat((cropped_images[2], cropped_images[3]), 2)),
                3
            )
            # patched_images = patched_images.to(device)

            data = torch.cat((patched_images.to(device), data), dim=0)

            output = model(data, dropout)

            eploss = sum([W_[k] * criteria(output[0:batch_size], c_[k]) for k in range(4)])
            acc = sum([W_[k] * accuracy(output[0:batch_size], c_[k])[0] for k in range(4)])

            eploss = eploss + criteria(output[batch_size:], target)

            pred = output[batch_size:].argmax(dim=1, keepdim=True)
            acc = acc.item() + (100 * pred.eq(target.view_as(pred)).sum().item() / batch_size)

            return eploss, acc/2

        model.train()
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            loss, batch_correct = ricap(data, target)

            if use_l1 == True:
                l1 = 0
                for p in model.parameters():
                    l1 = l1 + p.square().sum()
                loss = loss + lambda_l1 * l1
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            correct += batch_correct

        return epoch_loss / len(train_loader), correct / len(train_loader)

    return internal


def train(use_l1=False, lambda_l1=5e-4):
    """ Function to return train function instance

    Args:
        use_l1 (bool, optional): Enable L1. Defaults to False.
        lambda_l1 (float, optional): L1 Value. Defaults to 5e-4.
    """
    def internal(model, train_loader, optimizer, criteria, dropout, device, scheduler=None):
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


def get_sgd_optimizer(model, lr, momentum=0, weight_decay=0):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_nnl_criteria(device):
    return F.nll_loss


def get_crossentropy_criteria(device):
    return F.cross_entropy
