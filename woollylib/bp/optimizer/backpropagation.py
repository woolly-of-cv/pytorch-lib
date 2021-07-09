import torch
from torch.optim import SGD


torch.manual_seed(1)


def get_sgd_optimizer(model, lr, momentum=0, weight_decay=0):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
