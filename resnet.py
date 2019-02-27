from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math


__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert depth == 31, 'depth should be 31'

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_layer(block, 256, 1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.layer2 = self._make_layer(block, 256, 2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.layer3 = self._make_layer(block, 512, 5)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1,
                               bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1,
                               bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.layer2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        return x


# def resnet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return ResNet(**kwargs)


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
