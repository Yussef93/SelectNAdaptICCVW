# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pdb import set_trace as bp
from dassl.modeling.network.csg_builder import chunk_feature
import torch.utils.model_zoo as model_zoo
from dassl.modeling.backbone import BACKBONE_REGISTRY
from .lccs_module import LCCS


__all__ = ['ResNet', 'resnet101_lccs']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}
def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 ):
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
        #self.last = last
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=128, zero_init_residual=False,
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
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
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_features = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _idx in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                ))

        return nn.Sequential(*layers)

    def forward_fc(self, f4, task='old', f3=None, f2=None, return_mid_feature=False):
        x = f4
        if task in ['old', 'new']:
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
        if task == 'old':
            x = self.fc(x)
            return x
        else:
            if return_mid_feature:
                mid = self.fc_new[0](x)
                x = self.fc_new[1](mid)
                x = self.fc_new[2](x)
                return x, mid
            else:
                x = self.fc_new(x)
                return x

    def forward_partial(self, feature, stage):
        # stage: start forwarding **from** this stage (inclusive)
        # assert stage in [1, 2, 3, 4]
        if stage <= 1:
            feature = self.layer1(feature)
        if stage <= 2:
            feature = self.layer2(feature)
        if stage <= 3:
            feature = self.layer3(feature)
        if stage <= 4:
            feature = self.layer4(feature)
        return feature

    def forward_backbone(self, x, output_features=['layer4']):
        features = {}
        f0 = self.conv1(x)
        f0 = self.bn1(f0)
        f0 = self.relu(f0)
        if 'layer0' in output_features: features['layer0'] = f0
        f0 = self.maxpool(f0)
        f1 = self.layer1(f0)
        if 'layer1' in output_features: features['layer1'] = f1
        f2 = self.layer2(f1)
        if 'layer2' in output_features: features['layer2'] = f2
        f3 = self.layer3(f2)
        if 'layer3' in output_features: features['layer3'] = f3
        f4 = self.layer4(f3)
        if 'layer4' in output_features: features['layer4'] = f4
        if 'gap' in output_features:
            features['gap'] = self.avgpool(f4).view(f4.size(0), -1)
        return f4, features
        # return f4, f3, f2, features

    def forward(self, x, output_features=['layer4'], task='old'):
        '''
        task: 'old' | 'new' | 'new_seg'
        'old', 'new': classification tasks (ImageNet or Visda)
        'new_seg': segmentation head (convs)
        '''
        f4, features = self.forward_backbone(x, output_features)
        if 'fc_mid' in output_features:
            x, _mid = self.forward_fc(f4, task=task, return_mid_feature=True)
            features['fc_mid'] = _mid
        else:
            x = self.forward_fc(f4, task=task)
        return x, features

    def set_lccs_use_stats_status(self, status):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.set_use_stats_status(status)

    def set_lccs_update_stats_status(self, status):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.set_update_stats_status(status)

    def compute_source_stats(self):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.compute_source_stats()

    def set_svd_dim(self, svd_dim):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.set_svd_dim(svd_dim)

    def set_coeff(self, support_coeff_init, source_coeff_init):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.set_coeff(support_coeff_init, source_coeff_init)

    def initialize_trainable(self, support_coeff_init, source_coeff_init):
        for m in self.modules():
            if isinstance(m, LCCS):
                m.initialize_trainable(support_coeff_init, source_coeff_init)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, norm_layer=LCCS)
    if pretrained:
        from torch.utils.model_zoo import load_url
        state_dict = load_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in state and state[k].size() == v.size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    return model


@BACKBONE_REGISTRY.register()
def resnet101_lccs(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


