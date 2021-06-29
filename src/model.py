import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2


def get_norm_layer(output_size, norm="bn"):
    """This function provides normalization layer based on params

    Args:
        output_size (int): Number of output channel
        norm (str, optional): Parameter to decide which normalization to use,  Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.

    Returns:
       nn.Module : Instance of normalization class
    """
    n = nn.BatchNorm2d(output_size)
    if norm == "gn":
        n = nn.GroupNorm(GROUP_SIZE, output_size)
    elif norm == "ln":
        n = nn.GroupNorm(1, output_size)

    return n


class WyConv2d(nn.Module):
    """Creates an instance of 2d convolution based on differnet params provided


    Args:
        nn (nn.Module): Base Module class
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        strides=1,
        dilation=1,
        ctype="vanila",
        bias=False,
    ):
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
        if ctype == "depthwise":
            groups = in_channels
        elif ctype == "depthwise_seperable":
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
            bias=bias,
        )

        if ctype == "depthwise_seperable":
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=bias
            )

    def forward(self, x):
        x = self.conv(x)
        if self.ctype == "depthwise_seperable":
            x = self.pointwise(x)
        return x


class WyResidual(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        padding=1,
        strides=1,
        dilation=1,
        use1x1=False,
        ctype="vanila",
        norm="bn",
        first_block=False,
        usedilation=False,
    ):
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
                ctype=ctype,
            )
        else:
            self.conv1 = WyConv2d(
                input_size,
                output_size,
                kernel_size=3,
                padding=padding,
                strides=strides,
                dilation=dilation,
                ctype=ctype,
            )
        self.bn1 = get_norm_layer(output_size, norm=norm)
        self.conv2 = WyConv2d(
            output_size, output_size, kernel_size=3, padding=1, strides=1, ctype=ctype
        )
        self.bn2 = get_norm_layer(output_size, norm=norm)

        self.pointwise = None

        if use1x1:
            self.pointwise = WyConv2d(
                input_size, output_size, kernel_size=1, padding=0, strides=strides
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.pointwise:
            x = self.pointwise(x)

        if not self.first_block:
            y += x

        return F.relu(y)


class WyBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        repetations=2,
        ctype="vanila",
        norm="bn",
        padding=1,
        strides=2,
        dilation=1,
        use1x1=False,
        usepool=False,
        usedilation=False,
    ):
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
                    self.wyresudals.append(
                        WyResidual(
                            input_size,
                            output_size,
                            padding=0,
                            strides=strides,
                            dilation=dilation,
                            use1x1=use1x1,
                            ctype=ctype,
                            norm=norm,
                            usedilation=usedilation,
                        )
                    )
                else:
                    self.wyresudals.append(
                        WyResidual(
                            input_size,
                            output_size,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                            use1x1=use1x1,
                            ctype=ctype,
                            norm=norm,
                            usedilation=usedilation,
                        )
                    )
            else:
                self.wyresudals.append(
                    WyResidual(
                        output_size,
                        output_size,
                        padding=padding,
                        use1x1=use1x1,
                        ctype=ctype,
                        norm=norm,
                        usedilation=usedilation,
                    )
                )

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
    """Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(
        self,
        image,
        input_size=3,
        classes=10,
        base_channels=4,
        layers=3,
        drop_ratio=0.01,
        ctype="vanila",
        norm="bn",
        use1x1=False,
        usedilation=False,
    ):
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
        self.height, self.width = image
        self.dilation = 1

        super(WyCifar10Net, self).__init__()

        # Base Block
        self.b1 = WyResidual(input_size, self.base_channels * 2, first_block=True)

        # Transition + Residual Blocks 1
        if usedilation:
            self.dilation = (max(int(self.height / 4), 1), max(int(self.width / 4), 1))
        self.base_channels = self.base_channels * 2
        self.b2 = WyBlock(
            self.base_channels,
            self.base_channels * 2,
            repetations=self.layers,
            ctype=self.ctype,
            norm=self.norm,
            padding=1,
            dilation=self.dilation,
            use1x1=self.use1x1,
            usepool=False,
            usedilation=usedilation,
        )
        self.d2 = nn.Dropout(self.drop_ratio)
        self.height, self.width = self.height / 2, self.width / 2

        # Transition + Residual Block 2
        if usedilation:
            self.dilation = (max(int(self.height / 4), 1), max(int(self.width / 4), 1))
        self.base_channels = self.base_channels * 2
        self.b3 = WyBlock(
            self.base_channels,
            self.base_channels * 2,
            repetations=self.layers,
            ctype=self.ctype,
            norm=self.norm,
            padding=1,
            dilation=self.dilation,
            use1x1=self.use1x1,
            usepool=False,
            usedilation=usedilation,
        )
        self.height, self.width = self.height / 2, self.width / 2

        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(self.base_channels * 2, self.classes, 1)

    def forward(self, x, use_softmax=False, dropout=True):
        """Convolution function

        Args:
            x (tensor): Input image tensor
            dropout (bool, optional): Enable/Disable Dropout. Defaults to True.

        Returns:
            tensor: tensor of logits
        """

        # Input Layer
        x = self.b1(x)
        # Block 2
        x = self.b2(x)
        if dropout:
            x = self.d2(x)
        # Block 2
        x = self.b3(x)

        # Output Layer
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, self.classes)

        # Output Layer
        return F.log_softmax(x, dim=1) if use_softmax else x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
