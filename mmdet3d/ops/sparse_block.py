import torch
import torchsparse
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn
import torchsparse.nn as spnn
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck


class SparseBasicBlock(BasicBlock):
    """Sparse basic block for PartA^2.
    Sparse basic block implemented with submanifold sparse convolution.
    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the first block. Default: 1
        downsample (None | Module): down sample module for block.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        if act_cfg is not None:
            if act_cfg == "swish":
                self.relu = spnn.SiLU(inplace=True)
            else:
                self.relu = spnn.ReLU(inplace=True)


def make_sparse_convmodule(
    in_channels,
    out_channels,
    kernel_size,
    # indice_key,
    stride=1,
    padding=0,
    conv_type="TorchSparseConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
    activation_type="relu",
):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    conv_cfg = {"type": conv_type}

    layers = []
    for layer in order:
        if layer == "conv":
            layers.append(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
        elif layer == "norm":
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            if activation_type == "relu":
                layers.append(spnn.ReLU(inplace=True))
            elif activation_type == "swish":
               layers.append(spnn.SiLU(inplace=True))
            else:
                raise NotImplementedError
    layers = nn.Sequential(*layers)
    return layers
