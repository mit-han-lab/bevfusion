from typing import List, Tuple

import torch
from mmcv.cnn.resnet import BasicBlock, make_res_layer
from torch import nn

from mmdet.models import BACKBONES

__all__ = ["GeneralizedResNet"]


@BACKBONES.register_module()
class GeneralizedResNet(nn.ModuleList):
    def __init__(
        self,
        in_channels: int,
        blocks: List[Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks

        for num_blocks, out_channels, stride in self.blocks:
            blocks = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=1,
            )
            in_channels = out_channels
            self.append(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs
