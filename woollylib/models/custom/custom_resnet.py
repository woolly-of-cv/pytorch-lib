from matplotlib.pyplot import plasma
import torch.nn as nn
import torch.nn.functional as F

from woollylib.models.model import get_norm_layer, WyConv2d
from woollylib.models.transform.spatial_transformer import SpatialTransformer


class WyBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, norm, ctype, stride=1):
        super(WyBlock, self).__init__()

        # Preprocess Block
        self.pconv1 = WyConv2d(in_planes, planes, kernel_size=3,
                               strides=1, padding=1, ctype=ctype)
        self.pmp1 = nn.MaxPool2d(2)
        self.pbn1 = get_norm_layer(planes, norm)

        # ResBlock
        self.rconv1 = WyConv2d(planes, planes, kernel_size=3,
                               strides=stride, padding=1, ctype=ctype)
        self.rbn1 = get_norm_layer(planes, norm)
        self.rconv2 = WyConv2d(planes, planes, kernel_size=3,
                               strides=1, padding=1, ctype=ctype)
        self.rbn2 = get_norm_layer(planes, norm)

    def forward(self, x):
        x = F.relu(self.pbn1(self.pmp1(self.pconv1(x))))
        r = F.relu(self.rbn1(self.rconv1(x)))
        r = F.relu(self.rbn2(self.rconv2(r)))
        r = r + x
        return r + x


class View(nn.Module):
    def __init__(self, out_dim):
        super(View, self).__init__()
        self.out_dim = out_dim

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(-1, self.out_dim)

        return out


class CustomResNet(nn.Module):
    def __init__(self, block=WyBlock, ctype='vanila', norm='bn', classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes: int = 64
        self.classes: int = classes
        self.block: WyBlock = block

        # self.stn = SpatialTransformer()

        self.feature = nn.Sequential(
            self._pre_layer(norm, ctype='vanila'),
            self._res_layer(1, norm=norm, ctype=ctype, stride=1),
            self._transition_layer(norm=norm, ctype=ctype),
        )

        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(self.in_planes, classes, 1),
        #     View(self.classes)
        # )

        self.classifier = nn.Sequential(
            self._res_layer(1, norm=norm, ctype=ctype, stride=1),
            nn.MaxPool2d(4),  # nn.AdaptiveAvgPool2d(1),
            View(self.in_planes),
            nn.Linear(self.in_planes, self.classes)
        )

        # self.mp4 = nn.AdaptiveAvgPool2d(1) # nn.MaxPool2d(4)
        # self.linear = nn.Linear(self.in_planes, self.classes)

    def _pre_layer(self, norm, ctype):
        layer = nn.Sequential(
            WyConv2d(3, self.in_planes, kernel_size=3,
                     strides=1, padding=1, ctype=ctype),
            get_norm_layer(self.in_planes, norm),
            nn.ReLU()
        )

        return layer

    def _transition_layer(self, norm, ctype):
        out_planes = self.in_planes * self.block.expansion
        layer = nn.Sequential(
            WyConv2d(self.in_planes, out_planes,
                     kernel_size=3, strides=1, padding=1, ctype=ctype),
            nn.MaxPool2d(2),
            get_norm_layer(out_planes, norm),
            nn.ReLU()
        )

        self.in_planes = out_planes
        return layer

    def _res_layer(self, num_blocks, norm, ctype, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        out_planes = self.in_planes * self.block.expansion
        for stride in strides:
            layers.append(self.block(self.in_planes,
                          out_planes, norm, ctype, stride))
        self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, dropout=False):
        # stn network
        # x = self.stn(x)

        # Feature Layer
        out = self.feature(x)

        # Classifier Layer
        out = self.classifier(out)

        # out = self.mp4(out)
        # out = out.view(-1, self.in_planes)
        # out = self.linear(out)

        # Reshape
        # out = out.view(-1, self.classes)

        return out * 0.125
