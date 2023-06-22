
import torch

from mmdet3d.ops.feature_decorator import feature_decorator_ext


__all__ = ["feature_decorator"]

def feature_decorator(features, num_voxels, coords, vx, vy, x_offset, y_offset, normalize_coords, use_cluster, use_center):
    print(f'In feature decorator. Normalize: {normalize_coords}. Use cluster: {use_cluster}. Use center: {use_center}')

    # print(features.shape, num_voxels.shape, coords.shape)
    # print(features.dtype, num_voxels.dtype, coords.dtype)
    # print(features.device, num_voxels.device, coords.device)
    result = torch.ops.feature_decorator_ext.feature_decorator_forward(features, coords, num_voxels, vx, vy, x_offset, y_offset, normalize_coords, use_cluster, use_center)
    return result

if __name__ == '__main__':
    print('hi2')
    # print(torch.ops.pillar_pool_ext.pillar_pool_forward)
    # print(torch.ops.pillar_pool_ext.pillar_pool_backward)    
    # print(feature_decorator_ext.feature_decorator_forward)

    A = torch.ones((2, 20, 5), dtype=torch.float32).cuda()
    B = torch.ones(2, dtype=torch.int32).cuda()
    C = torch.ones((2, 4), dtype=torch.int32).cuda()

    D = feature_decorator(A, B, C)
    D = feature_decorator_ext.feature_decorator_forward(A, B, C)
    print(D)
    print(D.shape)

