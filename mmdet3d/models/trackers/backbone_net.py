import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstractionEdgeSA, PointNetFeaturePropagationSA

class Pointnet_Backbone(nn.Module):
    """
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 32, 32, 32],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=0.5,
                nsample=48,
                mlp=[64, 64, 64, 64],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=0.7,
                nsample=48,
                mlp=[128, 128, 128, 128],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[67,32,32], mlp_inte=[64, 3, 64, 64, 32]))
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[160,128,64], mlp_inte=[128, 32, 128, 64, 64]))    # 160=128+32
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[192,128,128], mlp_inte=[128, 64, 128, 64, 128]))  # 192=128+64


        self.cov_final = nn.Conv1d(32, 32, kernel_size=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)   # [B, N, 3], [B, C, N]
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        l_features[0] = xyz.transpose(1, 2).contiguous()
        for i in [2, 1, 0]:
            l_features[i] = self.FP_modules[i](l_xyz[i], l_xyz[i+1], l_features[i], l_features[i+1])

        return l_xyz[0], self.cov_final(l_features[0])  # [B, N, 3], [B, 32, N]