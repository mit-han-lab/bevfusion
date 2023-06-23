from .base_points import BasePoints
import torch

class RadarPoints(BasePoints):
    """Points of instances in LIDAR coordinates.
    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.
    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(RadarPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims
        )
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
            self.tensor[:, 4] = -self.tensor[:, 4]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]
            self.tensor[:, 3] = -self.tensor[:, 3]

    def jitter(self, amount):
        jitter_noise = torch.randn(self.tensor.shape[0], 3)
        jitter_noise *= amount 
        self.tensor[:, :3] += jitter_noise 

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.
        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor
        self.tensor[:, 3:5] *= scale_factor

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.
        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert (
            rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
        ), f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
                )
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                )
            elif axis == 0:
                rot_mat_T = rotation.new_tensor(
                    [[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]]
                )
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 3:5] = self.tensor[:, 3:5] @ rot_mat_T[:2, :2]

        return rot_mat_T

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.
        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).
        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.
        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.
        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        from mmdet3d.core.bbox import Coord3DMode

        return Coord3DMode.convert_point(point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)