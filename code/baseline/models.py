import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.distributions.multinomial import Multinomial


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def random_sample(cosine, sampling_num, temprature=0.5):
    batch_size, channels, h, w = cosine.shape
    prob = torch.exp(cosine / temprature)
    return Multinomial(sampling_num, prob.view(batch_size, channels, -1)).sample().view(cosine.shape)


class conv2d(nn.Conv2d):
    """nn.Conv2d + activation regularization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                     dilation, groups, bias, padding_mode)
        self.if_bias = bias

        # build a all-ones conv_kernel for computing norm of x
        self.all_one_conv_norm = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode)
        self.all_one_conv_norm.weight.data = torch.ones_like(self.weight, dtype=torch.float)
        self.all_one_conv_norm.weight.requires_grad = False

        # build another all-ones conv_kernel for computing sum of distances
        self.radius = 5  # radius of searching area
        self.all_one_conv_dist = nn.Conv2d(out_channels, out_channels, kernel_size=self.radius * 2 + 1,
                                           padding = self.radius, groups=out_channels, bias=False)
        self.all_one_conv_dist.weight.data = torch.ones_like(self.all_one_conv_dist.weight, dtype=torch.float)
        self.all_one_conv_dist.weight.requires_grad = False

    def forward(self, x):
        x_norm = self.all_one_conv_norm(x * x) ** 0.5
        x = super(conv2d, self).forward(x)

        if self.if_bias:
            cosine_activation = ((x - self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) /
                           (self.weight.norm(dim=-1).norm(dim=-1).norm(dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + 1e-6) /
                           (x_norm + 1e-6))
        else:
            cosine_activation = (x /
                           (self.weight.norm(dim=-1).norm(dim=-1).norm(dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + 1e-6) /
                           (x_norm + 1e-6))
        # print('!!!', cosine_activation.max(), cosine_activation.min())

        cosine_mean = self.all_one_conv_dist(cosine_activation) / ((self.radius * 2 + 1) ** 2)

        # print('***', cosine_mean.max(), cosine_mean.min())

        with torch.no_grad():
            random_mask = random_sample(cosine_activation, sampling_num=5)

        return x, ((random_mask * cosine_mean).sum(dim=(-2, -1)) / random_mask.sum(dim=(-2, -1))).mean()



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, if_return_activation=True):
    """1x1 convolution"""
    if if_return_activation:
        return conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        activation_cosine_sum = 0
        identity = x

        out, activation_cosine = self.conv1(x)
        activation_cosine_sum += activation_cosine
        out = self.bn1(out)
        out = self.relu(out)

        out, activation_cosine = self.conv2(out)
        activation_cosine_sum += activation_cosine
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, activation_cosine_sum


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        activation_cosine_sum = 0
        identity = x

        out, activation_cosine = self.conv1(x)
        activation_cosine_sum += activation_cosine
        out = self.bn1(out)
        out = self.relu(out)

        out, activation_cosine = self.conv2(out)
        activation_cosine_sum += activation_cosine
        out = self.bn2(out)
        out = self.relu(out)

        out, activation_cosine = self.conv3(out)
        activation_cosine_sum += activation_cosine
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, activation_cosine_sum


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, if_return_activation=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x, if_return_activation = False):
        activation_cosine_sum = 0

        x, activation_cosine = self.conv1(x)
        activation_cosine_sum += activation_cosine
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for module in self.layer1.children():
            x, activation_cosine = module(x)
            activation_cosine_sum += activation_cosine

        for module in self.layer2.children():
            x, activation_cosine = module(x)
            # activation_cosine_sum += activation_cosine

        for module in self.layer3.children():
            x, activation_cosine = module(x)
            # activation_cosine_sum += activation_cosine

        for module in self.layer4.children():
            x, activation_cosine = module(x)
            # activation_cosine_sum += activation_cosine


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if not if_return_activation:
            return x
        else:
            return x, activation_cosine_sum

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
