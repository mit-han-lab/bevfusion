# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from:
#    https://github.com/ucbdrive/dla/blob/master/dla.py
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

__all__ = ["DLA"]

class BasicBlock(nn.Module):
    def __init__(
        self, inplanes, planes, stride=1, dilation=1, 
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)

        out = out + residual
        out = F.relu_(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self, inplanes, planes, stride=1, dilation=1,
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = ConvModule(
            inplanes, 
            bottle_planes, 
            kernel_size=1, 
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv3 = ConvModule(
            bottle_planes, 
            planes, 
            kernel_size=1, 
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        out = out + residual
        out = F.relu_(out)

        return out


class Root(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, residual, 
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Root, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.residual = residual

    def forward(self, *x):
        children = x
        y = self.conv(torch.cat(x, 1))
        if self.residual:
            y = y + children[0]
        y = F.relu_(y)

        return y


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
        conv_cfg=dict(type="Conv2d"), 
        norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg, 
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg, 
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        # (dennis.park) If 'self.tree1' is a Tree (not BasicBlock), then the output of project is not used.
        # if in_channels != out_channels:
        if in_channels != out_channels and not isinstance(self.tree1, Tree):
            self.project = ConvModule(
                in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        # (dennis.park) If 'self.tree1' is a 'Tree', then 'residual' is not used.
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            y = self.root(x2, x1, *children)
        else:
            children.append(x1)
            y = self.tree2(x1, children=children)
        return y

@BACKBONES.register_module()
class DLA(BaseModule):
    def __init__(
        self,
        levels,
        channels,
        block=BasicBlock,
        residual_root=False,
        norm_eval=False,
        out_features=None,
        conv_cfg=dict(type="Conv2d"), 
        norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = ConvModule(
            3,
            channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU")
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level2 = Tree(
            levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level3 = Tree(
            levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level4 = Tree(
            levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level5 = Tree(
            levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.norm_eval = norm_eval

        if out_features is None:
            out_features = ['level5']
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        out_feature_channels, out_feature_strides = {}, {}
        for lvl in range(6):
            name = f"level{lvl}"
            out_feature_channels[name] = channels[lvl]
            out_feature_strides[name] = 2**lvl

        self._out_feature_channels = {name: out_feature_channels[name] for name in self._out_features}
        self._out_feature_strides = {name: out_feature_strides[name] for name in self._out_features}

    @property
    def size_divisibility(self):
        return 32

    def _make_conv_level(self, inplanes, planes, convs, conv_cfg, norm_cfg, act_cfg=None, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(
                ConvModule(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=norm_cfg is None,
                    dilation=dilation,
                    conv_cfg=conv_cfg, 
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="ReLU")
                )
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        assert x.dim() == 4, f"DLA takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.base_layer(x)
        for i in range(6):
            name = f"level{i}"
            x = self._modules[name](x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DLA, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
