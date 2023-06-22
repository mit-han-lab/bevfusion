from torch import nn

from typing import Any, Dict
from functools import cached_property

import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.resnet import make_res_layer, BasicBlock
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_backbone
from mmdet.models import BACKBONES
from torchvision.utils import save_image
from mmdet3d.ops import feature_decorator
from mmcv.cnn.bricks.non_local import NonLocal2d

from flash_attn.flash_attention import FlashMHA


__all__ = ["RadarFeatureNet", "RadarEncoder"]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer
        
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


@BACKBONES.register_module()
class RadarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "RadarFeatureNet"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels += 2
        # in_channels += 5
        self._with_distance = with_distance
        self.export_onnx = False

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        rfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            rfn_layers.append(
                RFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.rfn_layers = nn.ModuleList(rfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

    def forward(self, features, num_voxels, coors):
        # coors[:, 0] tells you the batch
        if not self.export_onnx:
            dtype = features.dtype

            # Find distance of x, y, and z from cluster center
            # features = features[:, :, :self.num_input]
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
                features
            ).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean

            f_center = torch.zeros_like(features[:, :, :2])
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
            )
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
            )

            # print(self.pc_range) [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] 
            # normalize x,y,z to [0, 1]
            features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            
            # Combine together feature decorations
            features_ls = [features, f_center]
            features = torch.cat(features_ls, dim=-1)

            # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
            # empty pillars remain set to zeros.
            voxel_count = features.shape[1]
            mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(features)
            features *= mask
            features = torch.nan_to_num(features)
        else:
            features = feature_decorator(features, num_voxels, coors, self.vx, self.vy, self.x_offset, self.y_offset, True, False, True)

        # Forward pass through PFNLayers
        for rfn in self.rfn_layers:
            features = rfn(features)

        return features.squeeze()


@BACKBONES.register_module()
class RadarEncoder(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_transformer_encoder=None, 
        pts_bev_encoder=None,
        post_scatter=None, 
        **kwargs,
    ):
        super().__init__()
        self.pts_voxel_encoder = build_backbone(pts_voxel_encoder)
        self.pts_middle_encoder = build_backbone(pts_middle_encoder)
        self.pts_transformer_encoder = build_backbone(pts_transformer_encoder) if pts_transformer_encoder is not None else None
        self.pts_bev_encoder = build_backbone(pts_bev_encoder) if pts_bev_encoder is not None else None
        self.post_scatter = build_backbone(post_scatter) if post_scatter is not None else None

        # if pts_middle_encoder.get('downsample', 1) > 1:
        #     assert pts_middle_encoder.get('downsample', 1) == 4
        #     channels = pts_middle_encoder['in_channels']
        #     self.post_scatter_conv = torch.nn.Sequential(
        #         nn.Conv2d(channels, channels, 3, stride=2, padding=1), 
        #         nn.Conv2d(channels, channels, 3, stride=2, padding=1), 
        #     )
        # elif pts_middle_encoder.get('share_conv', 0) > 0:
        #     channels = pts_middle_encoder['in_channels']
        #     conv_size = pts_middle_encoder['share_conv']

        #     norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01)
        #     conv_cfg=dict(type="Conv2d", bias=False)

        #     self.post_scatter_conv = torch.nn.Sequential(
        #         build_conv_layer(
        #             conv_cfg, channels, channels, conv_size, stride=1, padding=2, 
        #         ), 
        #         build_norm_layer(
        #             norm_cfg, channels
        #         )[1], 
        #         nn.ReLU(inplace=True), 
        #         build_conv_layer(
        #             conv_cfg, channels, channels, conv_size, stride=1, padding=2, 
        #         ), 
        #         build_norm_layer(
        #             norm_cfg, channels
        #         )[1], 
        #         nn.ReLU(inplace=True), 
        #         build_conv_layer(
        #             conv_cfg, channels, channels, conv_size, stride=1, padding=2, 
        #         ), 
        #         build_norm_layer(
        #             norm_cfg, channels
        #         )[1], 
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.post_scatter_conv = None

    def forward(self, feats, coords, batch_size, sizes, img_features=None):
        # self.visualize_pillars(feats, coords, sizes)
        # print(feats.shape)

        x = self.pts_voxel_encoder(feats, sizes, coords)
        # print(x.shape, coords.shape, batch_size)
        # return x
        if self.pts_transformer_encoder is not None:
            x = self.pts_transformer_encoder(x, sizes, coords, batch_size)

        x = self.pts_middle_encoder(x, coords, batch_size)

        if self.post_scatter is not None:
            x = self.post_scatter(x, img_features)
        
        if self.pts_bev_encoder is not None:
            x = self.pts_bev_encoder(x)
        
      
        # if x.shape[-1] != 128:
        #     x = F.interpolate(x, size=[128, 128])
        return x

    def visualize_pillars(self, feats, coords, sizes):
        nx, ny = 128, 128
        canvas = torch.zeros(
            nx*ny, dtype=sizes.dtype, device=sizes.device
        )
        indices = coords[:, 1] * ny + coords[:, 2]
        indices = indices.type(torch.long)
        canvas[indices] = sizes
        torch.save(canvas, 'sample_canvas')

@BACKBONES.register_module()
class PostScatterUNet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

@BACKBONES.register_module()
class PostScatterConvNet(nn.Module):

    def __init__(self, channels, num_layers=3, conv_size=5, padding=2, fuse_image=False):
        super().__init__()

        self.fuse_image = fuse_image
        block = []
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01)
        conv_cfg=dict(type="Conv2d", bias=False)
        for i in range(num_layers):
            block.append(
                make_res_layer(
                    BasicBlock, 
                    channels, 
                    channels, 
                    1, 
                    stride=1,
                    dilation=1,

                )
            )

            # Residual connections
            # block.append(
            #     build_conv_layer(
            #         conv_cfg, channels, channels, conv_size, stride=1, padding=padding, 
            #     )
            # )
            # block.append(build_norm_layer(norm_cfg, channels)[1])
            # block.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*block)

    def forward(self, x, img_bev_features=None):
        if self.fuse_image:
            if img_bev_features is None:
                raise ValueError("Post scatter conv net wants to fuse image features, but they weren't provided")
            x = x + img_bev_features # TODO I think concatenating would probably be better. 
        return self.net(x) 

@BACKBONES.register_module()
class PostScatterNonlocalNet(nn.Module):

    def __init__(self, channels, num_layers=3, fuse_image=False):
        super().__init__()

        self.fuse_image = fuse_image 
        block = []
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01)
        conv_cfg=dict(type="Conv2d")
        for i in range(num_layers):
            block.append(
                NonLocal2d(
                    channels, reduction=1, use_scale=False, conv_cfg=conv_cfg
                )
            )
            block.append(build_norm_layer(norm_cfg, channels)[1])
            block.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*block)

    def forward(self, x, img_bev_features=None):
        if self.fuse_image:
            if img_bev_features is None:
                raise ValueError("Post scatter conv net wants to fuse image features, but they weren't provided")
            x = x + img_bev_features # TODO I think concatenating would probably be better. 
        return self.net(x) 


@BACKBONES.register_module()
class TransformerEncoder(nn.Module):

    def __init__(self, num_blocks=3, num_heads=8, embed_dim=64, use_position=True, norm_before=False):
        super(TransformerEncoder, self).__init__()

        self.positional_embedding = PositionalEmbedding(pos_temperature = 10000, embed_dim=embed_dim) if use_position else None

        self.num_blocks = num_blocks 
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(TransformerEncoderBlock(num_heads, embed_dim, norm_before))
        # self._reset_parameters()

    def forward(self, x, sizes, coords, batch_size):
        # TODO positional embeddings 
        if self.positional_embedding is not None:
            pe = self.positional_embedding(x, coords)
            x = x + pe

        for block in self.blocks:
            x = block(x, sizes, coords, batch_size)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads=8, embed_dim=64, norm_before=False):
        super(TransformerEncoderBlock, self).__init__()

        self.mha = FlashMHA(
            embed_dim=embed_dim, # total channels (= num_heads * head_dim)
            num_heads=num_heads, # number of heads
        )
        activation = nn.ReLU
        # activation = nn.GeLU

        self.norm1 = nn.LayerNorm((embed_dim))
        self.norm2 = nn.LayerNorm((embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim), 
            activation(),
            nn.Linear(2*embed_dim, embed_dim), 
        )
        self.norm_before = norm_before


    def forward(self, x, sizes, coords, batch_size):
        # x should be something like (N, 64) here. 
        outputs = []

        for batch in range(batch_size):
            batch_mask = coords[:, 0] == batch 
            mha_input = x[batch_mask].unsqueeze(0)

            if self.norm_before:
                mha_output = self.mha(self.norm1(mha_input))[0][0]
                mlp_input = x[batch_mask] + mha_output
                mlp_output = self.mlp(self.norm2(mlp_input))
                output = mlp_output + mlp_input 
                outputs.append(output)
            else:
                mha_output = self.mha(mha_input)[0][0]
                mlp_input = self.norm1(x[batch_mask] + mha_output)
                mlp_output = self.mlp(mlp_input)
                output = self.norm2(mlp_input + mlp_output)
                outputs.append(output)

        outputs = torch.cat(outputs)
        return outputs

class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        # sparse_shape,
        # normalize_pos,
        pos_temperature,
        embed_dim,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # self.sparse_shape = sparse_shape
        # self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature

    def forward(self, x, coors):
        # size_x, size_y, size_z = self.sparse_shape
        x, y = coors[:, 1], coors[:, 2]

        # if self.normalize_pos:
        #     x = x / size_x * 2 * 3.1415  # [-pi, pi]
        #     y = y / size_y * 2 * 3.1415  # [-pi, pi]

        inv_freq = self.inv_freq

        # [num_tokens, pos_length]
        pex = x[:, None] / inv_freq[None, :]
        pey = y[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        pex = torch.stack([pex[:, ::2].sin(), pex[:, 1::2].cos()], dim=-1).flatten(1)
        pey = torch.stack([pey[:, ::2].sin(), pey[:, 1::2].cos()], dim=-1).flatten(1)
        pe = torch.cat([pex, pey], dim=-1).to(x.dtype)

        # gap = self.feat_dim - pe.size(1)
        # if gap > 0:
        #     pe_p = torch.zeros((pe.size(0), gap), dtype=dtype, device=coors.device)
        #     pe = torch.cat([pe, pe_p], dim=1)

        return pe

    @cached_property
    def inv_freq(self):
        ndim = 2
        pos_length = (self.embed_dim // (ndim * 2)) * 2

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device="cuda")
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        print(f'Inv freq shape: {inv_freq.shape}')
        return inv_freq

