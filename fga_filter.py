import numpy as np
from math import sqrt, atan2, cos, isnan


def _euclid_norm(vect):
    return sqrt(vect[0] * vect[0] + vect[1] * vect[1])


def _angle_rad(vect):
    return atan2(vect[1], vect[0])


def _grad(x, y, image):
    gradx = image[y][x - 1] - image[y][x + 1]
    grady = image[y + 1][x] - image[y - 1][x]
    return [gradx, grady]


def compute_grads_channel(image, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                grads[y, x] = _grad(x, y, image)


def compute_grads(image, grads):
    compute_grads_channel(image[:, :], grads[:, :, :])


def compute_modules_channel(image, modules, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                modules[y][x] = _euclid_norm(grads[y][x])


def compute_modules(image, modules, grads):
    compute_modules_channel(image[:, :], modules[:, :], grads[:, :, :])


def compute_angles_channel(image, angles, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                angle = _angle_rad(grads[y, x])
                if not isnan(angle):
                    angles[y, x] = angle
                else:
                    angles[y, x] = 0


def compute_angles(image, angles, grads):
    compute_angles_channel(image[:, :], angles[:, :], grads[:, :, :])


def smooth_channel(src, k_size, n=1, grads=None, modules=None, angles=None, dst=None):
    src_proxy = np.copy(src)
    if dst is None:
        dst = np.zeros(src.shape, np.float64)
    for i in range(n):
        if i == 0:
            _smooth_channel(src_proxy, k_size, grads=grads, modules=modules, angles=angles, dst=dst)
        else:
            _smooth_channel(src_proxy, k_size, dst=dst)
        src_proxy = dst
    if dst is not None:
        return dst


def _smooth_channel(src, k_size, grads=None, modules=None, angles=None, dst=None):
    if dst is None:
        dst = np.zeros(src.shape, dtype=np.float64)
    if grads is None:
        grads = np.zeros((src.shape[0], src.shape[1], 2))
        compute_grads_channel(src.astype(np.float64), grads)
    if modules is None:
        modules = np.zeros((src.shape[0], src.shape[1]))
        compute_modules_channel(src.astype(np.float64), modules, grads)
    if angles is None:
        angles = np.zeros((src.shape[0], src.shape[1]))
        compute_angles_channel(src.astype(np.float64), angles, grads)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            up = i - k_size // 2
            left = j - k_size // 2
            down = i + k_size // 2 + 1
            right = j + k_size // 2 + 1
            sum_weights = 0
            result = 0
            for s in range(up, down):
                if s < 0 or s >= src.shape[0]:
                    continue
                for t in range(left, right):
                    if t < 0 or t >= src.shape[1]:
                        continue
                    if modules[s][t] == .0:
                        continue

                    if s != i or t != j:
                        alpha = 1. / modules[s][t]
                        beta = 2. * (angles[i][j] - angles[s][t])
                        weight = (cos(beta) + 1) * alpha
                    else:
                        # weight of central pixel
                        weight = 1.
                    result += weight * src[s][t]
                    sum_weights += weight
            if sum_weights != 0:
                dst[i][j] = round(result / sum_weights)
            else:
                # pixel remains without changes if sum of weights = 0
                dst[i][j] = src[i][j]
    if dst is not None:
        return dst


def _smooth(src, dst, k_size, grads=None, modules=None, angles=None):
    if grads is None:
        grads = np.zeros((src.shape[0], src.shape[1], 2))
        compute_grads(src.astype(np.float64), grads)
    if modules is None:
        modules = np.zeros((src.shape[0], src.shape[1]))
        compute_modules(src.astype(np.float64), modules, grads)
    if angles is None:
        angles = np.zeros((src.shape[0], src.shape[1]))
        compute_angles(src.astype(np.float64), angles, grads)

    smooth_channel(src[:, :].astype(np.float64),
                   k_size,
                   grads=grads[:, :, :],
                   modules=modules[:, :],
                   angles=angles[:, :],
                   dst=dst[:, :])
    return dst


def smooth(src, k_size, n=1, grads=None, modules=None, angles=None):
    src_proxy = np.copy(src)
    dst = np.zeros(src.shape, np.float64)
    for i in range(n):
        if i == 0:
            _smooth(src_proxy, dst, k_size, grads=grads, modules=modules, angles=angles)
        else:
            _smooth(src_proxy, dst, k_size)
        src_proxy = dst
    return dst
