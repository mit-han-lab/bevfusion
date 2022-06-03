from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from mmdet.models import NECKS

__all__ = ["LSSFPN"]


@NECKS.register_module()
class LSSFPN(nn.Module):
    def __init__(
        self,
        in_indices: Tuple[int, int],
        in_channels: Tuple[int, int],
        out_channels: int,
        scale_factor: int = 1,
    ) -> None:
        super().__init__()
        self.in_indices = in_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x1 = x[self.in_indices[0]]
        assert x1.shape[1] == self.in_channels[0]

        x2 = x[self.in_indices[1]]
        assert x2.shape[1] == self.in_channels[1]

        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x1, x2], dim=1)

        x = self.fuse(x)
        if self.scale_factor > 1:
            x = self.upsample(x)
        return x
