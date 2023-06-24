from typing import Tuple

from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32

from torch import nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.builder import VTRANSFORMS
from mmdet.models.backbones.resnet import BasicBlock

from .base import BaseTransform, BaseDepthTransform

import torch

__all__ = ["AwareBEVDepth"]


class DepthRefinement(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthRefinement, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv_1 = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        self.depth_conv_2 = nn.Sequential(
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='Conv2d',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
            )),
            nn.BatchNorm2d(mid_channels), 
        )
        self.depth_conv_3 = nn.Sequential(
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(depth_channels), 
        )
        self.export = False

    def export_mode(self):
        self.export = True

    @force_fp32()
    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[1]
        ida = mats_dict['ida_mats'][:, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 4, 4).repeat(1, num_cams, 1, 1)

        # If exporting, cache the MLP input, since it's based on 
        # intrinsics and data augmentation, which are constant at inference time. 
        if not hasattr(self, 'mlp_input') or not self.export:
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            intrins[:, ..., 0, 0],
                            intrins[:, ..., 1, 1],
                            intrins[:, ..., 0, 2],
                            intrins[:, ..., 1, 2],
                            ida[:, ..., 0, 0],
                            ida[:, ..., 0, 1],
                            ida[:, ..., 0, 3],
                            ida[:, ..., 1, 0],
                            ida[:, ..., 1, 1],
                            ida[:, ..., 1, 3],
                            bda[:, ..., 0, 0],
                            bda[:, ..., 0, 1],
                            bda[:, ..., 1, 0],
                            bda[:, ..., 1, 1],
                            bda[:, ..., 2, 2],
                        ],
                        dim=-1,
                    ),
                    sensor2ego.view(batch_size, num_cams, -1),
                ],
                -1,
            )
            self.mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))


        x = self.reduce_conv(x)
        context_se = self.context_mlp(self.mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(self.mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv_1(depth) 
        depth = self.depth_conv_2(depth)
        depth = self.depth_conv_3(depth) 

        return torch.cat([depth, context], dim=1)


@VTRANSFORMS.register_module()
class AwareBEVDepth(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        use_points = 'lidar', 
        downsample: int = 1,
        bevdepth_downsample: int = 16, 
        bevdepth_refine: bool = True, 
        depth_loss_factor: float = 3.0, 
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            use_points=use_points,
        )
        self.depth_loss_factor = depth_loss_factor
        self.downsample_factor = bevdepth_downsample
        self.bevdepth_refine = bevdepth_refine
        if self.bevdepth_refine:
            self.refinement = DepthRefinement(self.C, self.C, self.C)

        self.depth_channels = self.frustum.shape[0]

        mid_channels = in_channels
        self.depthnet = DepthNet(
            in_channels, 
            mid_channels, 
            self.C, 
            self.D
        )


        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def export_mode(self):
        super().export_mode()
        self.depthnet.export_mode()

    @force_fp32()
    def get_cam_feats(self, x, mats_dict):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x, mats_dict)
        depth = x[:, : self.D].softmax(dim=1)

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        if self.bevdepth_refine:
            x = x.permute(0, 3, 1, 4, 2).contiguous() # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = x.shape
            x = x.view(-1, c, w, d)
            x = self.refinement(x)
            x = x.view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float()

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth


    def get_depth_loss(self, depth_labels, depth_preds):
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return self.depth_loss_factor * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()


    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x, depth_pred = x[0], x[-1]
        x = self.downsample(x)
        if kwargs.get('depth_loss', False):
            # print(kwargs['gt_depths'])
            depth_loss = self.get_depth_loss(kwargs['gt_depths'], depth_pred) 
            return x, depth_loss
        else:
            return x


@VTRANSFORMS.register_module()
class AwareDBEVDepth(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        use_points = 'lidar', 
        depth_input = 'scalar', 
        height_expand = False, 
        downsample: int = 1,
        bevdepth_downsample: int = 16, 
        bevdepth_refine: bool = True, 
        depth_loss_factor: float = 3.0, 
        add_depth_features = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            use_points=use_points,
            depth_input=depth_input,
            height_expand=height_expand,
            add_depth_features=add_depth_features,
        )
        self.depth_loss_factor = depth_loss_factor
        self.downsample_factor = bevdepth_downsample
        self.bevdepth_refine = bevdepth_refine
        if self.bevdepth_refine:
            self.refinement = DepthRefinement(self.C, self.C, self.C)

        self.depth_channels = self.frustum.shape[0]

        mid_channels = in_channels 
        self.depthnet = DepthNet(
            in_channels+64, 
            mid_channels, 
            self.C, 
            self.D
        )

        dtransform_in_channels = 1 if depth_input=='scalar' else self.D
        if self.add_depth_features:
            dtransform_in_channels += 45

        if depth_input == 'scalar':
            self.dtransform = nn.Sequential(
                nn.Conv2d(dtransform_in_channels, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        else:
            self.dtransform = nn.Sequential(
                nn.Conv2d(dtransform_in_channels, 32, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )


        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d, mats_dict):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x, mats_dict)
        depth = x[:, : self.D].softmax(dim=1)

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        if self.bevdepth_refine:
            x = x.permute(0, 3, 1, 4, 2).contiguous() # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = x.shape
            x = x.view(-1, c, w, d)
            x = self.refinement(x)
            x = x.view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float()

        # Here, x.shape is [num_cams, num_channels, depth_bins, downsampled_height, downsampled_width]
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth

    def export_mode(self):
        super().export_mode()
        self.depthnet.export_mode()

    def get_depth_loss(self, depth_labels, depth_preds):
        # if len(depth_labels.shape) == 5:
        #     # only key-frame will calculate depth loss
        #     depth_labels = depth_labels[:, 0, ...]

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return self.depth_loss_factor * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()


    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x, depth_pred = x[0], x[-1]
        x = self.downsample(x)
        if kwargs.get('depth_loss', False):
            depth_loss = self.get_depth_loss(kwargs['gt_depths'], depth_pred) 
            return x, depth_loss
        else:
            return x