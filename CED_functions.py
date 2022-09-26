import numpy as np
import math
from scipy import signal
from pathlib import Path
import skimage.io as io


def gaussian(img, fsize, sigma):   # fsize = filter size
    c = np.floor(fsize/2)
    G = np.zeros((fsize, fsize))
    for i in range(fsize):
        for j in range(fsize):
            G[i, j] = math.exp(-((i-c)**2+(j-c)**2)/(2*(sigma**2)))
    Gsum = np.sum(G)
    G = (1 / Gsum) * G
    g1 = signal.convolve2d(img, G, boundary='symm', mode='same')
    return g1


def bilateral_filter(Img: Path, k_size):
    img = io.imread(str(Img))
    M, N = img.shape
    sh = 10
    sg = 5
    G1 = np.zeros((k_size, k_size))

    for m in range(k_size):
        for n in range(k_size):
            G1[m, n] = np.exp(-(((m - k_size // 2) ** 2 + (n - k_size // 2) ** 2) / (2 * (sg ** 2))))
    kg = np.sum(G1)
    G = (1 / kg) * G1

    H1 = np.zeros(256)
    for m in range(256):
        H1[m] = np.exp(-(m**2) / (2 * (sh ** 2)))
    kh = np.sum(H1)
    H = (1/kh) * H1

    bimg = np.zeros(img.shape)
    for i in range(k_size//2, M - k_size//2):
        for j in range(k_size//2, N - k_size//2):
            temp = img[i - (k_size//2):i + (k_size//2) + 1, j - (k_size//2):j + (k_size//2) + 1]
            k = img[i, j] - temp
            a, b = k.shape
            Hm = np.zeros(k.shape)
            for m in range(a):
                for n in range(b):
                    Hm[m, n] = H[abs(k[m, n])]
            kij = np.sum(G * Hm)
            bimg[i, j] = (1/kij) * np.sum(G * Hm * temp)
    return bimg


def sobel(img):
    Mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    My = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    Gx = signal.convolve2d(img, Mx, boundary='symm', mode='same')
    Gy = signal.convolve2d(img, My, boundary='symm', mode='same')

    Mag_img = np.sqrt(np.square(Gx) + np.square(Gy))
    theta_img = np.arctan(Gy/Gx)
    theta_img = np.rad2deg(theta_img)
    theta_img = np.where(theta_img < 10000, theta_img, 0)

    # Mag_img = ((Mag_img - np.min(Mag_img)) / (np.max(Mag_img) - np.min(Mag_img))) * 255
    # Mag_img = Mag_img.astype(np.uint8)

    return Mag_img, theta_img


def prewitt(img):
    Mx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    My = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

    Gx = signal.convolve2d(img, Mx, boundary='symm', mode='same')
    Gy = signal.convolve2d(img, My, boundary='symm', mode='same')

    Mag_img = np.sqrt(np.square(Gx) + np.square(Gy))
    theta_img = np.arctan(Gy / Gx)
    theta_img = np.rad2deg(theta_img)
    theta_img = np.where(theta_img < 10000, theta_img, 0)

    return Mag_img, theta_img


def non_max_suppression(mag_img, theta_img, ksize):
    m, n = mag_img.shape
    x = ksize // 2
    for i in range(x, m-x):
        for j in range(x, n - x):
            if mag_img[i][j] != 0:
                temp_arr = np.zeros(ksize)
                stp = ksize // 2

                # sector = 0
                if -22.5 <= theta_img[i][j] <= 22.5:
                    for t in range(ksize):
                        temp_arr[t] = mag_img[i][j - stp]
                        stp = stp - 1

                # sector = 1
                elif 22.5 < theta_img[i][j] <= 67.5:
                    for t in range(ksize):
                        temp_arr[t] = mag_img[i + stp][j - stp]
                        stp = stp - 1

                # sector = 2
                elif theta_img[i][j] < -67.5 or theta_img[i][j] > 67.5:
                    for t in range(ksize):
                        temp_arr[t] = mag_img[i - stp][j]
                        stp = stp - 1

                # sector = 3
                elif -67.5 <= theta_img[i][j] < -22.5:
                    for t in range(ksize):
                        temp_arr[t] = mag_img[i - stp][j - stp]
                        stp = stp - 1

                if mag_img[i][j] != np.max(temp_arr):
                    mag_img[i][j] = 0

    return mag_img


def non_max_suppression2(mag_img, theta_img):
    m, n = mag_img.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if mag_img[i][j] != 0:

                # sector = 0
                if -22.5 <= theta_img[i][j] <= 22.5:
                    if (mag_img[i, j] >= mag_img[i, j - 1]) and (mag_img[i, j] >= mag_img[i, j + 1]):
                        mag_img[i, j - 1] = 0
                        mag_img[i, j + 1] = 0
                    else:
                        mag_img[i, j] = 0

                # sector = 1
                elif 22.5 < theta_img[i][j] <= 67.5:
                    if (mag_img[i, j] >= mag_img[i + 1, j - 1]) and (mag_img[i, j] >= mag_img[i - 1, j + 1]):
                        mag_img[i + 1, j - 1] = 0
                        mag_img[i - 1, j + 1] = 0
                    else:
                        mag_img[i, j] = 0

                # sector = 2
                elif theta_img[i][j] < -67.5 or theta_img[i][j] > 67.5:
                    if (mag_img[i, j] >= mag_img[i - 1, j]) and (mag_img[i, j] >= mag_img[i + 1, j]):
                        mag_img[i - 1, j] = 0
                        mag_img[i + 1, j] = 0
                    else:
                        mag_img[i, j] = 0

                # sector = 3
                elif -67.5 <= theta_img[i][j] < -22.5:
                    if (mag_img[i, j] >= mag_img[i - 1, j - 1]) and (mag_img[i, j] >= mag_img[i + 1, j + 1]):
                        mag_img[i - 1, j - 1] = 0
                        mag_img[i + 1, j + 1] = 0
                    else:
                        mag_img[i, j] = 0

    return mag_img


def Hysteresis_Thresholding(img, t_low, t_high):
    img1 = np.where(img > t_high, 255, 0)
    img2 = np.where(img > t_low, 255, 0)
    m, n = img.shape
    for i in range(1, m-1):
        for j in range(1, n - 1):
            if (img1[i, j] == 0) and ((img1[i, j - 1] == 255) and (img1[i, j+1] == 255)):
                if img2[i, j] == 255:
                    img1[i, j] = 255
            elif (img1[i, j] == 0) and ((img1[i - 1, j] == 255) and (img1[i + 1, j] == 255)):
                if img2[i, j] == 255:
                    img1[i, j] = 255
            elif (img1[i, j] == 0) and ((img1[i - 1, j - 1] == 255) and (img1[i + 1, j + 1] == 255)):
                if img2[i, j] == 255:
                    img1[i, j] = 255
            elif (img1[i, j] == 0) and ((img1[i - 1, j + 1] == 255) and (img1[i + 1, j - 1] == 255)):
                if img2[i, j] == 255:
                    img1[i, j] = 255
    return img1
