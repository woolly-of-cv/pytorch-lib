import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

class GradCAM(nn.Module):
    def __init__(self, model):
        super(GradCAM, self).__init__()
        
        # get the pretrained network
        self.wy = model
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.wy.feature
        
        # get the classifier of the model
        self.classifier = self.wy.classifier
        
        # placeholder for the gradients
        self.gradients = None
        
        self.classes = self.wy.classes
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.classifier(x)

        x = x.view(-1, self.classes)
        
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)