import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import transforms as tr
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
POINT_TARGET = np.array([[
    -0.3,
    0.3,
    0.7,
    1
]]).T
THETA = np.array([0, -np.pi/4, np.pi/2])
LENGTH = np.array([0, 0.55, 0.30])
TRAJECTORY_Z_MIDPOINT = 0.6

PROJECTION_1 = camera_matrix([np.pi/2, 0, 0], [0, -1, 0.5])
PROJECTION_2 = camera_matrix([np.pi/2, 0, -np.pi/2], [-1, 0, 0.5])
PROJECTIONS = np.stack((PROJECTION_1, PROJECTION_2))


def forward_kinematics(theta: np.ndarray):
    x_1 = POINT_ORIGIN
    H_1 = tr.rotation_matrix([0, 0, theta[0]])
    H_2 = tr.homogenous_matrix([0, theta[1], 0], [0, 0, LENGTH[0]])
    H = H_1 @ H_2
    x_3 = H @ POINT_ORIGIN
    H_3 = tr.homogenous_matrix([0, theta[2], 0], [LENGTH[1], 0, 0])
    H = H @ H_3
    x_4 = H @ POINT_ORIGIN
    H_4 = tr.translation_matrix([LENGTH[2], 0, 0])
    H = H @ H_4
    x_5 = H @ POINT_ORIGIN
    X = np.column_stack((x_1, x_3, x_4, x_5))
    return X


def render(X, target, POINT_TARGET, img_X, img_target, img_TARGET):
    fig = plt.figure(figsize=(12, 9))
    ax_3d = fig.add_subplot(2, 2, (1, 2), projection='3d')
    ax_image_left = fig.add_subplot(2, 2, 3)
    ax_image_right = fig.add_subplot(2, 2, 4)
    axs = (ax_3d, ax_image_left, ax_image_right)
    ax_3d, ax_image_left, ax_image_right = axs
    ax_image_left.sharex(ax_image_right)
    render_images(img_X, img_target, img_TARGET, axs[1:])

    ax_3d.plot(X[0, :], X[1, :], X[2, :], 'o-', label='Robot')
    ax_3d.plot(POINT_TARGET[0], POINT_TARGET[1], POINT_TARGET[2], 'o', label='Target')
    ax_3d.plot(target[0], target[1], target[2], 'o', label='Trajectory Target')
    ax_3d.plot(X[0, -1], X[1, -1], X[2, -1], 'o', label='End Effector')
    ax_3d.legend(loc='upper right')
    ax_3d.set_xlabel('X')
    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylabel('Y')
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlabel('Z')
    ax_3d.set_zlim(0, 1)
    ax_3d.set_aspect('equal')


def render_images(img_X, img_target, img_TARGET, axs):
    img_end_effector = img_X[:, :, -1]
    for i, ax in enumerate(axs):
        ax.plot(img_X[i, 0], img_X[i, 1], 'o-', label='Robot')
        ax.plot(img_TARGET[i, 0], img_TARGET[i, 1], 'o', label='Target')
        ax.plot(img_target[i, 0], img_target[i, 1], 'o', label='Trajectory Target')
        ax.plot(img_end_effector[i, 0], img_end_effector[i, 1], 'o', label='End Effector')
        ax.set_aspect('equal')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)


def img_error(img_points):
    end_effector = img_points[:2, -3]
    target = img_points[:2, -1]
    return end_effector - target


def init_jacobian_central_differences(theta, target, delta=0.15):
    J = np.zeros((4, 3))
    for i in range(3):
        theta[i] += delta
        X = forward_kinematics(theta)
        img_X = project_points(PROJECTIONS, X)
        img_end_effector = img_X[:, :, -1]
        img_target = project_points(PROJECTIONS, target)
        e_r = get_error(img_end_effector, img_target)
        theta[i] -= 2 * delta
        X = forward_kinematics(theta)
        img_X = project_points(PROJECTIONS, X)
        img_end_effector = img_X[:, :, -1]
        img_target = project_points(PROJECTIONS, target)
        e_l = get_error(img_end_effector, img_target)
        J[:, i] = (e_r - e_l) / (2 * delta)
        theta[i] += delta
    return J


def generate_trajectory(start, stop, num=10):
    stop = np.squeeze(stop)
    trajectory = np.linspace(start, stop, num).T
    # fit a quadratic polynomial on z
    poly = np.polyfit([-1, 0, 1], [start[2], TRAJECTORY_Z_MIDPOINT, stop[2]], 2)
    trajectory[2, :] = np.polyval(poly, np.linspace(-1, 1, num))
    return trajectory


def main():
    theta = THETA
    X = forward_kinematics(theta)
    img_X = project_points(PROJECTIONS, X)
    img_TARGET = project_points(PROJECTIONS, POINT_TARGET)
    render(X, POINT_TARGET, POINT_TARGET, img_X, img_TARGET, img_TARGET)
    plt.show()
    plt.close()

    trajectory = generate_trajectory(X[:, -1], POINT_TARGET)
    for target in trajectory.T:
        theta = broydens_method(theta, target)

    X = forward_kinematics(theta)
    img_X = project_points(PROJECTIONS, X)
    img_target = project_points(PROJECTIONS, target)
    img_TARGET = project_points(PROJECTIONS, POINT_TARGET)
    render(X, target, POINT_TARGET, img_X, img_target, img_TARGET)
    plt.show()


def broydens_method(theta, target):
    B = init_jacobian_central_differences(theta, target)
    img_TARGET = project_points(PROJECTIONS, POINT_TARGET)
    e_prev = None
    s_prev = None
    for _ in range(N):
        X = forward_kinematics(theta)
        img_X = project_points(PROJECTIONS, X)
        img_end_effector = img_X[:, :, -1]
        img_target = project_points(PROJECTIONS, target)
        e = get_error(img_end_effector, img_target)
        if la.norm(e) < 1e-3:
            plt.close()
            break
        s = la.lstsq(B, -e, rcond=None)[0]
        theta += alpha * s
        if e_prev is not None:
            y = e - e_prev
            B += beta * np.outer(y - B @ s_prev, s_prev.T) / (s_prev.T @ s_prev)
        e_prev = e
        s_prev = s
        render(X, target, POINT_TARGET, img_X, img_target, img_TARGET)
        plt.pause(0.01)
        plt.close()
    return theta


def project_points(Ps, points):
    if len(points.shape) < 2:
        points = points.reshape((-1, 1))
    image = Ps @ points
    image = np.transpose(image, (1, 0, 2))
    image = image / image[2, :, :]
    image = np.transpose(image, (1, 0, 2))
    return image


def get_error(img_end_effector, img_target):
    error = (img_end_effector[:, :2] - img_target.squeeze()[:, :2]).flatten()
    return error


if __name__ == '__main__':
    main()
