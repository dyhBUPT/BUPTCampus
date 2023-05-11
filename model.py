"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/30 15:57
@Discription: model
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from resnet import resnet34, resnet50, resnet101, remove_fc, model_urls
from utils import *


def get_model(opt, class_num=1, name='Baseline'):
    if name == 'Baseline':
        model = Baseline(
            class_num=class_num,
            backbone=opt.backbone,
            temporal=opt.temporal,
            one_stream=opt.one_stream,
        )
    model.cuda()
    if opt.gpu_mode == 'dp':
        model = nn.DataParallel(model)
    return model


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        return x / norm


class BottleNeck(nn.Module):
    def __init__(self, feat_dim):
        super(BottleNeck, self).__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)  # no shiftgi
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bn(x)


class Classifier(nn.Module):
    def __init__(self, feat_dim, class_num, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        return self.fc(x)


class modality_speficic_module(nn.Module):
    FLAG = False # 加载整个backbone
    def __init__(self, backbone='resnet50', input_channel=3):
        super(modality_speficic_module, self).__init__()
        pretrained = input_channel == 3
        if self.FLAG:
            self.backbone = eval(backbone)(
                pretrained=pretrained,
                last_conv_stride=1,
                last_conv_dilation=1,
                input_channel=input_channel,
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            if pretrained:
                state_dict = remove_fc(model_zoo.load_url(model_urls[backbone]))
                self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if self.FLAG:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        return x


class modality_shared_module(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(modality_shared_module, self).__init__()
        self.backbone = eval(backbone)(
            pretrained=True,
            last_conv_stride=1,
            last_conv_dilation=1
        )

    def forward(self, x):
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


class temporal_module(nn.Module):
    def __init__(self, method='gap', feat_dim=2048):
        super(temporal_module, self).__init__()
        self.method = method
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.gmp = nn.AdaptiveMaxPool1d(output_size=1)
        if method == 'self-attention':
            self.transformer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu'
            )

    def forward(self, x):
        """
        :param x: shape [b,t,c]
        :return: shape [b,c]
        """
        b, t, c = x.size()
        if self.method == 'gap':
            x = x.permute(0, 2, 1)
            x = self.gap(x)
        elif self.method == 'gmp':
            x = x.permute(0, 2, 1)
            x = self.gmp(x)
        elif self.method == 'self-attention':
            x = x + self.transformer(x)
            x = x.permute(0, 2, 1)
            x = self.gap(x)
        x = x.view(b, -1)
        return x


class Baseline(nn.Module):
    def __init__(self, class_num, backbone='resnet50', temporal='gap', one_stream=False):
        super(Baseline, self).__init__()
        if backbone in ['resnet18', 'resnet34']:
            feat_dim = 512
        elif backbone in ['resnet50', 'resnet101', 'resnet152']:
            feat_dim = 2048
        else:
            raise RuntimeError('Wrong backbone.')
        self.one_stream = one_stream
        self.shared_module = modality_shared_module(backbone)
        self.ir_module = modality_speficic_module(backbone, 3)
        self.rgb_module = modality_speficic_module(backbone, 3)
        self.classifier = Classifier(feat_dim, class_num, bias=False)
        self.temporal_module = temporal_module(temporal, feat_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = BottleNeck(feat_dim)
        self.l2norm = Normalize(2)

    def forward(self, x_rgb=None, x_ir=None, pids=None):
        # self.rgb_module = self.ir_module
        # [b,t,c,h,w]
        if x_rgb is not None and x_ir is not None:
            assert x_rgb.size() == x_ir.size()
            b, t, c, h, w = x_rgb.size()
            x_rgb = x_rgb.contiguous().view(-1, c, h, w)
            x_ir = x_ir.contiguous().view(-1, c, h, w)
            if self.one_stream:
                x_rgb = self.rgb_module(x_rgb)
                x_ir = self.rgb_module(x_ir)
            else:
                x_rgb = self.rgb_module(x_rgb)
                x_ir = self.ir_module(x_ir)

            x = torch.cat((x_rgb, x_ir), dim=0)

        elif x_rgb is not None:
            b, t, c, h, w = x_rgb.size()
            x_rgb = x_rgb.view(-1, c, h, w)
            x = self.rgb_module(x_rgb)
        elif x_ir is not None:
            b, t, c, h, w = x_ir.size()
            x_ir = x_ir.view(-1, c, h, w)
            if self.one_stream:
                x = self.rgb_module(x_ir)
            else:
                x = self.ir_module(x_ir)
        else:
            raise RuntimeError('Both x_rgb and x_ir are None.')

        x = self.shared_module(x)  # [bt,c,h,w] e.g., [160,2048,16,8]
        features = self.avgpool(x).squeeze()  # [bt,c]
        features = features.view(features.size(0)//t, t, -1)  # [b,t,c]
        features = self.temporal_module(features)  # [b,c]
        features_bn = self.bottleneck(features)

        if self.training:
            pids = pids.repeat(2)
            return features, self.classifier(features_bn), pids
        else:
            return self.l2norm(features_bn)
