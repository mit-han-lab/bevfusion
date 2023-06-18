import itertools

from mmcv.cnn.bricks.registry import CONV_LAYERS, NORM_LAYERS
from torch.nn.parameter import Parameter

def register_torchsparse():
    """This func registers torchsparse ops."""
    from torchsparse.nn import Conv3d, BatchNorm

    CONV_LAYERS._register_module(Conv3d, "TorchSparseConv3d", force=True)
    NORM_LAYERS._register_module(BatchNorm, "TorchSparseBatchNorm", force=True)

register_torchsparse()
