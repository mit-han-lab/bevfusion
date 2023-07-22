
import torch

from . import feature_decorator_ext


__all__ = ["feature_decorator"]

def feature_decorator(features, num_voxels, coords, vx, vy, x_offset, y_offset, normalize_coords, use_cluster, use_center):
    result = torch.ops.feature_decorator_ext.feature_decorator_forward(features, coords, num_voxels, vx, vy, x_offset, y_offset, normalize_coords, use_cluster, use_center)
    return result

if __name__ == '__main__':

    A = torch.ones((2, 20, 5), dtype=torch.float32).cuda()
    B = torch.ones(2, dtype=torch.int32).cuda()
    C = torch.ones((2, 4), dtype=torch.int32).cuda()
    D = feature_decorator(A, B, C)
    D = feature_decorator_ext.feature_decorator_forward(A, B, C)
    print(D.shape)

