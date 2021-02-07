import torch.nn as nn
import torch


# 18和34层的残差结构
class BasicBlock(nn.Module):
    # residual block中主分支的卷积核的个数有没有发生变化，若卷积核个数一模一样，就将系数设为1
    expansion = 1

    # in_channel: 输入特征矩阵的深度
    # out_channel: 输出特征矩阵的深度，对应的就是主分支上卷积核的个数
    # stride: 步长默认取1
    # downsample: 默认取None，有时候需要降维
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False) # stride = 1, 实现连接(1*1不会改变特征矩阵的大小)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 50层，101层和152层的残差结构
class Bottleneck(nn.Module):
    # 残差结构中，第三层卷积层的卷积核个数是第一层和第二层的4倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # ------------
        # squeeze channels
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # ------------
        # unsqueeze channels
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # block: 对应的残差结构，18和34层网络结构使用BasicBlock, 50和101和152层使用Bottleneck
    # block_num: 一个残差结构数量的列表，如34层的就是[3 4 6 3]
    # num_classes: 训练集分类的个数
    # include_top: 方便未来在ResNet的基础上搭建更加复杂的网络
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        # 这里的in_channel是通过max_pooling之后得到的特征矩阵的深度，无论是18层、34层、50层、101层还是152层，对应的输出深度都是64
        self.in_channel = 64

        # in_channels = 3 是因为输入图像为RGB图像
        # 这一系列的操作就是为了让输出的特征矩阵的高和宽变为原来的一半：112 -> 56
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.in_channel:
            self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))    # 自适应的，无论输入的是什么形状大小，输出都是1*1
            # 因为输出之后大小是1*1，展平之后的大小就是特征矩阵的深度了
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    # block: BasicBlock or Bottleneck
    # channel: residual block中卷积层使用的卷积核个数，这里对应的是第一层卷积层的个数（18和34都是64；50和101和152都是有四倍关系）
    # block_num: 该层总共包含了多少个残差结构。例如ResNet34中的conv2_x有三个残差结构
    # stride:
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 对于18和34的ResNet会直接跳过这个if语句，50和101和152则不会
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 1*1
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel), channel)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)