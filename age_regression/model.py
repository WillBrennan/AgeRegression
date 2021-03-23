import torch
import torch.nn as nn
from torchvision import models


def channel_shuffle(x, g: int):
    n, _, h, w = x.shape

    x = x.reshape(n, g, -1, h, w)
    x = x.permute(0, 2, 1, 3, 4)

    x = x.reshape(n, -1, h, w)

    return x


class SALayer(nn.Module):
    def __init__(self, c: int, g: int = 64):
        super().__init__()
        self.c = c
        self.g = g

        k = c // (2 * g)

        self.cw = nn.Parameter(torch.zeros(1, k, 1, 1))
        self.cb = nn.Parameter(torch.zeros_like(self.cw))
        self.sw = nn.Parameter(torch.zeros_like(self.cw))
        self.sb = nn.Parameter(torch.zeros_like(self.cw))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(k, k)

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.reshape(n * self.g, -1, h, w)

        x_0, x_1 = x.chunk(2, dim=1)

        xn = self.avg_pool(x_0)
        xn = xn * self.cw + self.cb
        xn = x_0 * torch.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sw * xs + self.sb
        xs = x_1 * torch.sigmoid(xs)

        out = torch.cat((xs, xn), dim=1)
        out = out.reshape(n, -1, h, w)

        out = channel_shuffle(out, 2)
        return out


# note(will.brennan) - from torchvision resnet but with the SALayer at the end
class SABottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.sa_layer = SALayer(planes * 4)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.sa_layer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def sa_resnet50(pretrained: bool = False):
    return models.ResNet(SABottleneck, [3, 4, 6, 3])


def sa_resnet101(pretrained: bool = False):
    return models.ResNet(SABottleneck, [3, 4, 23, 3])


class AgeRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = sa_resnet101()
        self.model.fc = nn.Linear(512 * 4, 1)

    def forward(self, x: torch.Tensor):
        x_age = self.model(x)
        return x_age
