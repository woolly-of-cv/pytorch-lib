'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from woollylib.models.model import get_norm_layer, WyConv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm, ctype, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = WyConv2d(in_planes, planes, kernel_size=3,
                              strides=stride, padding=1, ctype=ctype)
        self.bn1 = get_norm_layer(planes, norm)  # nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv2 = WyConv2d(planes, planes, kernel_size=3,
                              strides=1, padding=1, ctype=ctype)
        self.bn2 = get_norm_layer(planes, norm)  # nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes,
                #           kernel_size=1, stride=stride, bias=False),
                WyConv2d(in_planes, self.expansion*planes, kernel_size=1,
                         strides=stride, padding=0, ctype='vanila'),
                # nn.BatchNorm2d(self.expansion*planes)
                get_norm_layer(self.expansion*planes, norm)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, ctype, norm, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        # Feature Layer
        # self.first = self.first_block(norm, ctype)
        # self.layer1 = self._make_layer(
        #     block, 64, num_blocks[0], norm=norm, ctype=ctype, stride=1)
        # self.layer2 = self._make_layer(
        #     block, 128, num_blocks[1], norm=norm, ctype=ctype, stride=2)
        # self.layer3 = self._make_layer(
        #     block, 256, num_blocks[2], norm=norm, ctype=ctype, stride=2)

        self.feature = nn.Sequential(
            self._first_layer(norm, ctype='vanila'),
            self._make_layer(
                block, 64, num_blocks[0], norm=norm, ctype=ctype, stride=1),
            self._make_layer(
                block, 128, num_blocks[1], norm=norm, ctype=ctype, stride=2),
            self._make_layer(
                block, 256, num_blocks[2], norm=norm, ctype=ctype, stride=2)
        )
        # Classifier Layer
        # self.layer4 = self._make_layer(
        #     block, 512, num_blocks[3], norm=norm, ctype=ctype, stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        self.classifier = nn.Sequential(
            self._make_layer(
                block, 512, num_blocks[3], norm=norm, ctype=ctype, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, num_classes, 1)
        )

    def _first_layer(self, norm, ctype):
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        return nn.Sequential(
            WyConv2d(3, 64, kernel_size=3,
                     strides=1, padding=1, ctype=ctype),
            get_norm_layer(64, norm),  # nn.BatchNorm2d(64)
            nn.ReLU()
        )

    def _make_layer(self, block, planes, num_blocks, norm, ctype, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm, ctype, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        # Feature Layer
        out = self.feature(x)

        # Classifier Layer
        out = self.classifier(out)

        # Reshape
        out = out.view(-1, self.num_classes)

        return out


def ResNet18(ctype='vanila', norm='bn', num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], ctype=ctype, norm=norm, num_classes=num_classes)


def ResNet34(ctype='vanila', norm='bn', num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], ctype=ctype, norm=norm, num_classes=num_classes)


def test():
    net = ResNet18(ctype='vanila', norm='bn', num_classes=10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
