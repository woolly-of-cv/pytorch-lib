import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2


def get_norm_layer(output_size, norm='bn'):
    """This function provides normalization layer based on params

    Args:
        output_size (int): Number of output channel
        norm (str, optional): Parameter to decide which normalization to use,  Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.

    Returns:
       nn.Module : Instance of normalization class
    """
    n = nn.BatchNorm2d(output_size)
    if norm == 'gn':
        n = nn.GroupNorm(GROUP_SIZE, output_size)
    elif norm == 'ln':
        n = nn.GroupNorm(1, output_size)

    return n


class WyConv2d(nn.Module):
    """Creates an instance of 2d convolution based on differnet params provided


    Args:
        nn (nn.Module): Base Module class
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1, dilation=1, ctype='vanila', bias=False):
        """Init Custom calss

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size to be used in convolution. Defaults to 3.
            ctype (str, optional): Type of convolution to be used. Allowed values ['vanila', 'depthwise', 'depthwise_seperable'] Defaults to 'vanila'.
            bias (bool, optional): Enable/Disable Bias. Defaults to False.
        """
        super(WyConv2d, self).__init__()
        self.ctype = ctype
        groups = 1
        out = out_channels
        if ctype == 'depthwise':
            groups = in_channels
        elif ctype == 'depthwise_seperable':
            groups = in_channels
            out = in_channels

        self.conv = nn.Conv2d(in_channels, out, kernel_size=kernel_size,
                              stride=strides, groups=groups, padding=padding, dilation=dilation, bias=bias)

        if ctype == 'depthwise_seperable':
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.ctype == 'depthwise_seperable':
            x = self.pointwise(x)
        return x


class WyResidual(nn.Module):
    def __init__(self, input_size, output_size, padding=1, strides=1, dilation=1, use1x1=False, ctype='vanila', norm='bn', first_block=False, usedilation=False, use_skip=True):
        super(WyResidual, self).__init__()

        self.first_block = first_block

        if usedilation:
            self.conv1 = WyConv2d(input_size, output_size, kernel_size=3,
                                  padding=padding, strides=1, dilation=dilation, ctype=ctype)
        else:
            self.conv1 = WyConv2d(input_size, output_size, kernel_size=3,
                                  padding=padding, strides=strides, dilation=dilation, ctype=ctype)
        self.bn1 = get_norm_layer(output_size, norm=norm)
        self.conv2 = WyConv2d(
            output_size, output_size, kernel_size=3, padding=1, strides=1, ctype=ctype)
        self.bn2 = get_norm_layer(output_size, norm=norm)

        self.pointwise = None
        self.use_skip = use_skip

        if use1x1 and use_skip:
            self.pointwise = WyConv2d(
                input_size, output_size, kernel_size=1, padding=0, strides=strides)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.pointwise:
            x = self.pointwise(x)

        if not self.first_block and self.use_skip:
            y += x

        return F.relu(y)


class WyBlock(nn.Module):
    def __init__(self, input_size, output_size, repetations=2, ctype='vanila', norm='bn', padding=1, strides=2, dilation=1, use1x1=False, usepool=False, usedilation=False, use_skip=True):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
            usepool (bool, optional): Enable/Disable Maxpolling. Defaults to True.
        """
        super(WyBlock, self).__init__()
        self.usepool = usepool
        self.wyresudals = []
        for r in range(repetations):
            if r == 0:
                if usedilation:
                    self.wyresudals.append(WyResidual(
                        input_size, output_size, padding=0, strides=strides, dilation=dilation, use1x1=use1x1, ctype=ctype, norm=norm, usedilation=usedilation, use_skip=use_skip))
                else:
                    self.wyresudals.append(WyResidual(
                        input_size, output_size, padding=padding, strides=strides, dilation=dilation, use1x1=use1x1, ctype=ctype, norm=norm, usedilation=usedilation, use_skip=use_skip))
            else:
                self.wyresudals.append(WyResidual(
                    output_size, output_size, padding=padding, use1x1=use1x1, ctype=ctype, norm=norm, usedilation=usedilation, use_skip=use_skip))

        self.conv = nn.Sequential(*self.wyresudals)

        if usepool:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block
            layers (int, optional): Number of layers in this block. Defaults to 3.
            last (bool, optional): Is this the last block. Defaults to False.

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv(x)

        if self.usepool:
            x = self.pool(x)

        return x


class WyCifar10Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, image, input_size=3, classes=10, base_channels=4, layers=3, drop_ratio=0.01, ctype='vanila', norm='bn', use1x1=False, usedilation=False, use_skip=True):
        """Initialize Network

        Args:
            base_channels (int, optional): Number of base channels to start with. Defaults to 4.
            layers (int, optional): Number of Layers in each block. Defaults to 3.
            drop (float, optional): Dropout value. Defaults to 0.01.
            norm (str, optional): Normalization type. Defaults to 'bn'.
        """
        # Variables
        self.input_size = input_size
        self.classes = classes
        self.base_channels = base_channels
        self.layers = layers
        self.drop_ratio = drop_ratio
        self.ctype = ctype
        self.norm = norm
        self.use1x1 = use1x1
        self.use_skip = use_skip
        self.height, self.width = image
        self.dilation = 1

        self.blocks = []

        super(WyCifar10Net, self).__init__()

        # Base Block
        self.blocks.append(WyResidual(
            input_size, self.base_channels*2, first_block=True))

        # Transition + Residual Blocks 1
        if usedilation:
            self.dilation = (max(int(self.height/4), 1),
                             max(int(self.width/4), 1))
        self.base_channels = self.base_channels*2
        self.blocks.append(WyBlock(self.base_channels, self.base_channels*2, repetations=self.layers,
                                   ctype=self.ctype, norm=self.norm, padding=1, dilation=self.dilation, use1x1=self.use1x1, usepool=False, usedilation=usedilation, use_skip=self.use_skip))
        # self.d2 = nn.Dropout(self.drop_ratio)
        self.height, self.width = self.height/2, self.width/2

        # Transition + Residual Block 2
        if usedilation:
            self.dilation = (max(int(self.height/4), 1),
                             max(int(self.width/4), 1))
        self.base_channels = self.base_channels*2
        self.blocks.append(WyBlock(self.base_channels, self.base_channels*2, repetations=self.layers,
                                   ctype=self.ctype, norm=self.norm, padding=1, dilation=self.dilation, use1x1=self.use1x1, usepool=False, usedilation=usedilation, use_skip=self.use_skip))
        self.height, self.width = self.height/2, self.width/2

        # Combine Feature Layer
        self.feature = nn.Sequential(*self.blocks)

        # Output Block
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.flat = nn.Conv2d(self.base_channels*2, self.classes, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.base_channels*2, self.classes, 1)
        )

    def forward(self, x, use_softmax=False, dropout=True):
        """Convolution function

        Args:
            x (tensor): Input image tensor
            dropout (bool, optional): Enable/Disable Dropout. Defaults to True.

        Returns:
            tensor: tensor of logits
        """

        # Feature Layer
        x = self.feature(x)

        # Classifier Layer
        x = self.classifier(x)

        # Reshape
        x = x.view(-1, self.classes)

        # Output Layer
        return F.log_softmax(x, dim=1) if use_softmax else x
