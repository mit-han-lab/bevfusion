from typing import Any
import numpy as np
try:
    from numpy.typing import ArrayLike
except:
    from typing import Union
    ArrayLike = Union[tuple, list, np.ndarray]
from scipy.spatial.transform import Rotation as R


def rotate2d(x: ArrayLike, angle: float) -> np.ndarray:
    """Rotate an array of 2D points/vectors counterclockwise

    :param x: an (N, 2) array representing N 2D points/vectors
    :type x: array_like
    :param angle: the angle to be rotated in radian
    :type angle: float
    :returns: an (N, 2) np.ndarray with the points/vectors rotated
    :rtype: np.ndarray
    """

    sin, cos = np.sin(angle), np.cos(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    return np.dot(np.asarray(x)[:,:2], rot.T)


def affine_transform(
    matrix: ArrayLike = None,
    scale: ArrayLike = None,
    rotation: Any = None,
    rotation_format: str = 'quat',
    translation: ArrayLike = None,
    dim=3
) -> np.ndarray: 
    # TODO: update docstring
    # TODO: add error checking and prompts

    batch_size = 0

    if matrix is None:
        matrix = np.eye(dim)
    else:
        matrix = np.asarray(matrix)
        if batch_size == 0 and matrix.ndim == 3:
            batch_size = matrix.shape[0]

    if scale is not None:
        scale = np.asarray(scale)
        scale_mat = np.eye(dim)
        if batch_size == 0 and scale.ndim == 2:
            batch_size = scale.shape[0]
            scale_mat = np.tile(scale_mat, (batch_size,1,1))
        scale_mat[...,np.arange(dim),np.arange(dim)] *= scale
        matrix = scale_mat @ matrix

    if rotation is not None:
        if dim == 2:
            rotmat = R.from_euler('z', rotation).as_matrix()[...,:2,:2]
        elif dim == 3:
            if rotation_format == 'quat':
                rotmat = R.from_quat(rotation).as_matrix()
            elif rotation_format == 'euler':
                rotmat = R.from_euler(*rotation).as_matrix()
            elif rotation_format == 'rotvec':
                rotmat = R.from_rotvec(rotation).as_matrix()
            elif rotation_format == 'rotmat':
                rotmat = np.asarray(rotation)
            else:
                raise ValueError(f'invalid 3D rotation format {rotation_format}')
        elif dim > 3:
            if rotation_format == 'rotmat':
                rotmat = np.asarray(rotation)
            else:
                raise ValueError(f'invalid {dim}D rotation format {rotation_format}')
        if batch_size == 0 and rotmat.ndim == 3:
            batch_size = rotmat.shape[0]
        matrix = rotmat @ matrix

    if translation is None:
        translation = np.zeros(dim)
    else:
        translation = np.asarray(translation)
        if batch_size == 0 and translation.ndim == 2:
            batch_size = translation.shape[0]

    # Construct homogeneous transformation matrix
    hmat = np.eye(dim+1)
    if batch_size > 0:
        hmat = np.tile(hmat, (batch_size,1,1))
    hmat[...,:dim,:dim] = matrix
    hmat[...,:dim,dim] = translation
    return hmat


def apply_transform(hmat: ArrayLike, x: ArrayLike) -> np.ndarray:
    # TODO: update docstring

    hmat = np.asarray(hmat)
    x = np.asarray(x)
    return ((hmat[...,:3,:3] @ x[...,:3,None])[...,0] + hmat[...,:3,3]) / \
        (np.sum(hmat[...,3,:3]*x, axis=-1) + hmat[...,3,3])[...,None]
