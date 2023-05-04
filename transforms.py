import numpy as np


def rotation_matrix(rotation: np.ndarray):
    """
    Compute the rotation matrix from rotation vector.

    Args:
        rotation (3 * 1): rotation vector

    Returns:
        R (4 * 4): rotation matrix
    """
    R = np.eye(4)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotation[0]), -np.sin(rotation[0])],
        [0, np.sin(rotation[0]), np.cos(rotation[0])]
    ])
    Ry = np.array([
        [np.cos(rotation[1]), 0, np.sin(rotation[1])],
        [0, 1, 0],
        [-np.sin(rotation[1]), 0, np.cos(rotation[1])]
    ])
    Rz = np.array([
        [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
        [np.sin(rotation[2]), np.cos(rotation[2]), 0],
        [0, 0, 1]
    ])
    R[:3, :3] = Rz @ Ry @ Rx
    return R


def translation_matrix(translation: np.ndarray):
    t = np.array(translation).reshape(-1, 1)
    H = np.eye(4)
    H[:3, 3:] = t
    return H


def homogenous_matrix(rotation: np.ndarray, translation: np.ndarray):
    R = rotation_matrix(rotation)[:3, :3]
    H = translation_matrix(translation)
    H[:3, :3] = R
    return H
