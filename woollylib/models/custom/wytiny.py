import torch
import torch.nn as nn
import torch.nn.functional as F

from woollylib.models.common import View

torch.manual_seed(1)

GROUP_SIZE = 16


class GhostBatchNorm2d(nn.Module):
    def __init__(self, inp, vbs=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(vbs, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, x.size(0)//self.vbs, 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


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
    elif norm == 'gbn':
        n = GhostBatchNorm2d(output_size)

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

        self.conv = nn.Conv2d(
            in_channels,
            out,
            kernel_size=kernel_size,
            stride=strides,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        if ctype == 'depthwise_seperable':
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias
            )

    def forward(self, x):
        x = self.conv(x)
        if self.ctype == 'depthwise_seperable':
            x = self.pointwise(x)
        return x


class WyBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            input_size,
            output_size,
            padding=1,
            strides=1,
            dilation=1,
            use1x1=False,
            ctype='vanila',
            norm='bn',
            first_block=False,
            usedilation=False,
            use_skip=True):

        super(WyBottleneck, self).__init__()
        planes = int(output_size/self.expansion)
        self.conv1 = WyConv2d(
            input_size,
            planes,
            kernel_size=1,
            padding=0,
            strides=1
        )
        self.bn1 = get_norm_layer(planes, norm=norm)

        self.conv2 = WyConv2d(
            planes,
            planes,
            kernel_size=3,
            padding=padding,
            strides=strides,
            dilation=dilation,
            ctype=ctype
        )
        self.bn2 = get_norm_layer(planes, norm=norm)

        self.conv3 = WyConv2d(
            planes,
            output_size,
            kernel_size=1,
            padding=0,
            strides=1
        )

        self.bn3 = get_norm_layer(output_size, norm=norm)

        self.shortcut = nn.Sequential()
        if use1x1 and use_skip and strides != 1 and input_size != output_size:
            self.shortcut = nn.Sequential(
                WyConv2d(
                    input_size,
                    output_size,
                    kernel_size=1,
                    padding=0,
                    strides=strides
                ),
                get_norm_layer(output_size, norm=norm)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WyResidual(nn.Module):
    expansion = 2

    def __init__(self, input_size, output_size, padding=1, strides=1, dilation=1, use1x1=False, ctype='vanila', norm='bn', first_block=False, usedilation=False, use_skip=True):
        super(WyResidual, self).__init__()

        self.first_block = first_block

        if usedilation:
            self.conv1 = WyConv2d(
                input_size,
                output_size,
                kernel_size=3,
                padding=padding,
                strides=1,
                dilation=dilation,
                ctype=ctype
            )
        else:
            self.conv1 = WyConv2d(
                input_size,
                output_size,
                kernel_size=3,
                padding=padding,
                strides=strides,
                dilation=dilation,
                ctype=ctype
            )
        self.bn1 = get_norm_layer(output_size, norm=norm)
        self.conv2 = WyConv2d(
            output_size,
            output_size,
            kernel_size=3,
            padding=1,
            strides=1,
            ctype=ctype
        )
        self.bn2 = get_norm_layer(output_size, norm=norm)

        self.pointwise = nn.Sequential()
        self.use_skip = use_skip

        if use1x1 and use_skip and strides != 1 and input_size != output_size:
            self.pointwise = nn.Sequential(
                WyConv2d(
                    input_size,
                    output_size,
                    kernel_size=1,
                    padding=0,
                    strides=strides
                ),
                get_norm_layer(output_size, norm=norm)
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.pointwise:
            x = self.pointwise(x)

        if not self.first_block and self.use_skip:
            y += x

        return F.relu(y)


class WyBlock(nn.Module):
    def __init__(self, input_size, output_size, repetations=2, ctype='vanila', norm='bn', padding=1, strides=2, dilation=1, use1x1=False, usepool=False, usedilation=False, use_skip=True, block=WyResidual):
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
                    self.wyresudals.append(block(
                        input_size, output_size, padding=0, strides=strides, dilation=dilation, use1x1=use1x1, ctype=ctype, norm=norm, usedilation=usedilation, use_skip=use_skip))
                else:
                    self.wyresudals.append(block(
                        input_size, output_size, padding=padding, strides=strides, dilation=dilation, use1x1=use1x1, ctype=ctype, norm=norm, usedilation=usedilation, use_skip=use_skip))
            else:
                self.wyresudals.append(block(
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
        if self.usepool:
            x = self.pool(x)

        x = self.conv(x)

        return x


class WyTiny(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, image, input_size=3, classes=10, base_channels=4, layers=3, drop_ratio=0.01, ctype='vanila', norm='bn', use1x1=False, usedilation=False, use_skip=True, blocks_count=2):
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
        self.blocks_count = blocks_count
        self.resblock = WyResidual

        self.blocks = []

        super(WyTiny, self).__init__()

        # Base Block
        self.blocks.append(WyResidual(
            input_size, self.base_channels*2, first_block=True))

#         self.blocks.append(self._pre_layer(input_size))

        self.base_channels = self.base_channels*self.resblock.expansion

        for i, s in [(2, 1), (2, 1), (2, 1)]:  # range(self.blocks_count):
            inc, outc = self.base_channels, self.base_channels*self.resblock.expansion
            self.blocks.append(self._bottle_neck(inc, outc))
            self.blocks.append(self._make_block(
                usedilation, i, s, resblock=self.resblock))
            self.base_channels = self.base_channels*self.resblock.expansion

        # Combine Feature Layer
        self.feature = nn.Sequential(*self.blocks)

        # Output Block
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.flat = nn.Conv2d(self.base_channels*2, self.classes, 1)
        self.classifier = nn.Sequential(
            #             nn.Conv2d(self.base_channels, self.classes, 1),
            nn.AdaptiveAvgPool2d(1),
            #             View(self.classes),
            View(self.base_channels),
            nn.Linear(self.base_channels, self.classes)
        )

#     def _pre_layer(self, input_size):
#         return nn.Sequential(
#             nn.ZeroPad2d(3),
#             nn.Conv2d(input_size, self.base_channels*2, 7, 2, bias=False),
#             nn.BatchNorm2d(self.base_channels*2),
#             nn.ReLU(),
# #             nn.MaxPool2d(3, 2)
#         )

    def _bottle_neck(self, inp, output):
        return nn.Sequential(
            WyConv2d(inp, output),
            nn.MaxPool2d(2),
            get_norm_layer(output, norm=self.norm),
            nn.ReLU()
        )

    def _make_block(self, usedilation, repetations, strides, resblock):
        if usedilation:
            self.dilation = (
                max(int(self.height/4), 1),
                max(int(self.width/4), 1)
            )
        if strides != 1:
            inc, outc = self.base_channels, self.base_channels*resblock.expansion
        else:
            inc, outc = self.base_channels * \
                resblock.expansion, self.base_channels*resblock.expansion
        block = WyBlock(
            inc,
            outc,
            repetations=repetations,
            ctype=self.ctype,
            norm=self.norm,
            padding=1,
            strides=strides,
            dilation=self.dilation,
            use1x1=self.use1x1,
            usepool=False,
            usedilation=usedilation,
            use_skip=self.use_skip,
            block=resblock
        )
        self.height, self.width = self.height/2, self.width/2

        return block

    def forward(self, x, use_softmax=False, dropout=False):
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
#         x = x.view(-1, self.classes)

        # Output Layer
        return F.log_softmax(x, dim=1) if use_softmax else x
