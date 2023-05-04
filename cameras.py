import numpy as np
import transforms as tr


def camera_matrix(rotation: np.ndarray, translation: np.ndarray):
    P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1/3, 1]
    ])
    R_c = tr.rotation_matrix(rotation)[:3, :3]
    t_c = np.array(translation).reshape(-1, 1)
    E = np.eye(4)
    E[:3, :3] = R_c.T
    E[:3, 3:] = -R_c.T @ t_c
    P = P @ E
    return P
