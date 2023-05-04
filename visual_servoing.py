import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from cameras import camera_matrix

N = 20
alpha = 0.1
beta = 0.5
POINT_ORIGIN = np.array([
    0,
    0,
    0,
    1
]).T
POINT_TARGET = np.array([
    -0.3,
    0.3,
    0.3,
    1
]).T
THETA = np.array([0, -np.pi/4, np.pi/2])
LENGTH = np.array([0, 0.55, 0.30])
TRAJECTORY_Z_MIDPOINT = 0.6

PROJECTION = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
ROTATION_CAMERA_1 = np.array([np.pi/2, 0, 0])
TRANSLATION_CAMERA_1 = np.array([0, -1, 0.5])
PROJECTION_1 = camera_matrix(ROTATION_CAMERA_1, TRANSLATION_CAMERA_1)

PROJECTION_2 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
ROTATION_CAMERA_2 = np.array([
    np.pi/2, 0, -np.pi/2]
)
TRANSLATION_CAMERA_2 = np.array([-1, 0, 0.5])
PROJECTION_2 = camera_matrix(ROTATION_CAMERA_2, TRANSLATION_CAMERA_2)


def forward_kinematics(theta: np.ndarray):
    x_1 = POINT_ORIGIN
    H_1 = np.array([
        [np.cos(theta[0]), -np.sin(theta[0]), 0, 0],
        [np.sin(theta[0]), np.cos(theta[0]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    H_2 = np.array([
        [np.cos(theta[1]), 0, np.sin(theta[1]), 0],
        [0, 1, 0, 0],
        [-np.sin(theta[1]), 0, np.cos(theta[1]), LENGTH[0]],
        [0, 0, 0, 1]
    ])
    x_3 = H_1 @ H_2 @ POINT_ORIGIN
    H_3 = np.array([
        [np.cos(theta[2]), 0, np.sin(theta[2]), LENGTH[1]],
        [0, 1, 0, 0],
        [-np.sin(theta[2]), 0, np.cos(theta[2]), 0],
        [0, 0, 0, 1]
    ])
    x_4 = H_1 @ H_2 @ H_3 @ POINT_ORIGIN
    H_4 = np.array([
        [1, 0, 0, LENGTH[2]],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    x_5 = H_1 @ H_2 @ H_3 @ H_4 @ POINT_ORIGIN
    X = np.column_stack((x_1, x_3, x_4, x_5))
    return X


def render(X, p, p2, img_points_left, img_points_right):
    fig = plt.figure(figsize=(12.8, 9.6))
    ax_3d = fig.add_subplot(2, 2, (1, 2), projection='3d')
    ax_image_left = fig.add_subplot(2, 2, 3)
    ax_image_right = fig.add_subplot(2, 2, 4)
    axs = (ax_3d, ax_image_left, ax_image_right)
    ax_3d, ax_image_left, ax_image_right = axs
    ax_image_left.sharex(ax_image_right)
    render_image(img_points_left, ax_image_left)
    render_image(img_points_right, ax_image_right)

    ax_3d.plot(X[0, :], X[1, :], X[2, :], 'o-', label='Robot')
    ax_3d.plot(p[0], p[1], p[2], 'o', label='Target')
    ax_3d.plot(p2[0], p2[1], p2[2], 'o', label='Trajectory Target')
    ax_3d.legend(loc='upper right')
    ax_3d.plot(X[0, -1], X[1, -1], X[2, -1], 'o-', label='End Effector')
    ax_3d.set_xlabel('X')
    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylabel('Y')
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlabel('Z')
    ax_3d.set_zlim(0, 1)
    ax_3d.set_aspect('equal')


def render_image(img_points, ax_image):
    ax_image.plot(img_points[0, :-2], img_points[1, :-2], 'o-', label='Robot')
    ax_image.plot(img_points[0, -1], img_points[1, -1], 'o', label='Target')
    ax_image.plot(img_points[0, -2], img_points[1, -2], 'o', label='Trajectory Target')
    ax_image.plot(img_points[0, -3], img_points[1, -3], 'o-', label='End Effector')
    ax_image.set_aspect('equal')
    ax_image.set_xlabel('u')
    ax_image.set_ylabel('v')
    ax_image.set_xlim(-1, 1)
    ax_image.set_ylim(-1, 1)


def img_error(img_points):
    end_effector = img_points[:2, -3]
    target = img_points[:2, -1]
    return end_effector - target


def init_jacobian_central_differences(theta, target, delta=0.15):
    J = np.zeros((4, 3))
    for i in range(3):
        theta[i] += delta
        _, img_points_left, img_points_right = get_points(theta, target)
        e_r = get_error(img_points_left, img_points_right)
        theta[i] -= 2 * delta
        _, img_points_left, img_points_right = get_points(theta, target)
        e_l = get_error(img_points_left, img_points_right)
        J[:, i] = (e_r - e_l) / (2 * delta)
        theta[i] += delta
    return J


def generate_trajectory(start, stop, num=10):
    trajectory = np.linspace(start, stop, num).T
    # fit a quadratic polynomial on z
    poly = np.polyfit([-1, 0, 1], [start[2], TRAJECTORY_Z_MIDPOINT, stop[2]], 2)
    trajectory[2, :] = np.polyval(poly, np.linspace(-1, 1, num))
    return trajectory


def main():
    theta = THETA
    X, img_points_left, img_points_right = get_points(theta, POINT_TARGET)
    render(X, POINT_TARGET, POINT_TARGET, img_points_left, img_points_right)
    plt.show()
    plt.close()

    trajectory = generate_trajectory(X[:, -1], POINT_TARGET)
    for target in trajectory.T:
        X, img_points_left, img_points_right, theta = broydens_method(theta, target)

    render(X, POINT_TARGET, POINT_TARGET, img_points_left, img_points_right)
    plt.show()


def broydens_method(theta, target):
    B = init_jacobian_central_differences(theta, target)
    e_prev = None
    s_prev = None
    for _ in range(N):
        X, img_points_left, img_points_right = get_points(theta, target)
        e = get_error(img_points_left, img_points_right)
        if la.norm(e) < 1e-3:
            plt.close()
            break
        s = la.lstsq(B, -e, rcond=None)[0]
        theta += alpha * s
        if e_prev is not None:
            y = e - e_prev
            B += beta * np.outer(y - B @ s_prev, s_prev.T) / (s_prev.T @ s_prev)
            # J = init_jacobian_central_differences(theta, target)
            # print(f"J: {J}")
            # print(f"B: {B}")
            # B = init_jacobian_central_differences(theta)
        e_prev = e
        s_prev = s
        render(X, POINT_TARGET, target, img_points_left, img_points_right)
        plt.pause(0.01)
        plt.close()
    return X, img_points_left, img_points_right, theta


def get_points(theta, target):
    X = forward_kinematics(theta)
    points = np.column_stack((X, POINT_TARGET, target))
    img_points_left = PROJECTION_1 @ points
    img_points_right = PROJECTION_2 @ points
    return X, img_points_left, img_points_right


def get_error(img_points_left, img_points_right):
    error_left = img_error(img_points_left)
    error_right = img_error(img_points_right)
    error = np.hstack((error_left, error_right))
    return error


if __name__ == '__main__':
    main()
