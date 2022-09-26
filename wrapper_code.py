from pathlib import Path
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from data.src.CED_functions import gaussian, sobel, prewitt, non_max_suppression, non_max_suppression2, Hysteresis_Thresholding, bilateral_filter
import fga_filter as fga


def canny_edge_detection():
    img_path = Path('../images/noisy_book2.png')
    # img_path = Path('../images/Bee.jpg')
    img = io.imread(img_path.as_posix())
    # img = color.rgb2gray(img) * 255

    # Step-1  ---->  smoothing
    # # Gaussian
    # img_low = gaussian(img, fsize=3, sigma=5)  # fsize = filter size

    # # bilateral
    # img_low = bilateral_filter(img_path, 3)

    # filter based on gradient
    kernel_size = 3  # set kernel_size = 3 for filtering with 3x3 kernel
    runs_number = 1  # set number of runs: parameter n is 1 by default
    img_low = fga.smooth(img, kernel_size, n=runs_number)  # smooth image

    # Using sobel
    # Step-2  ----> gradient magnitude and direction
    Mag_img1, theta_img1 = sobel(img_low)

    # Step-3  ----> non maximum suppression
    Mag_img_after_nms1 = non_max_suppression(Mag_img1, theta_img1, ksize=3)
    # Mag_img_after_nms1 = non_max_suppression2(Mag_img1, theta_img1)

    # Step-4  -----> hysteresis thresholding
    t_low = 60
    t_high = 4 * t_low
    CED_img1 = Hysteresis_Thresholding(Mag_img_after_nms1, t_low, t_high)

    # Using prewitt
    # Step-2  ----> gradient magnitude and direction
    Mag_img2, theta_img2 = prewitt(img_low)

    # Step-3  ----> non maximum suppression
    Mag_img_after_nms2 = non_max_suppression(Mag_img2, theta_img2, ksize=3)
    # Mag_img_after_nms2 = non_max_suppression2(Mag_img2, theta_img2)

    # Step-4  -----> hysteresis thresholding
    t_low = 60
    t_high = 4 * t_low
    CED_img2 = Hysteresis_Thresholding(Mag_img_after_nms2, t_low, t_high)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title(f"original image")
    # plt.subplot(221)
    # plt.imshow(img_low, 'gray')
    # plt.title(f"gaussian low pass with \nwindow size 5x5, sigma = 1")
    plt.subplot(222)
    plt.imshow(Mag_img1, 'gray')
    plt.title(f"gradient magnitude image using sobel")
    plt.subplot(223)
    plt.imshow(Mag_img_after_nms1, 'gray')
    plt.title(f"gradient magnitude image using sobel after nms")
    plt.subplot(224)
    plt.imshow(CED_img1, 'gray')
    plt.title(f"edge detection using sobel with canny edge detector")

    plt.figure()
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title(f"original image")
    # plt.subplot(221)
    # plt.imshow(img_low, 'gray')
    # plt.title(f"gaussian low pass with \nwindow size 5x5, sigma = 1")
    plt.subplot(222)
    plt.imshow(Mag_img2, 'gray')
    plt.title(f"gradient magnitude image using prewitt")
    plt.subplot(223)
    plt.imshow(Mag_img_after_nms2, 'gray')
    plt.title(f"gradient magnitude image using prewitt after nms")
    plt.subplot(224)
    plt.imshow(CED_img2, 'gray')
    plt.title(f"edge detection using prewitt with canny edge detector")

    plt.show()
    pass


def smoothing_algos():
    img_path = Path('../images/noisy_book2.png')
    img = io.imread(img_path.as_posix())

    # Gaussian
    img_low = gaussian(img, fsize=3, sigma=5)  # fsize = filter size

    # bilateral
    img_low1 = bilateral_filter(img_path, 3)

    # filter based on gradient
    kernel_size = 3  # set kernel_size = 3 for filtering with 3x3 kernel
    runs_number = 1  # set number of runs: parameter n is 1 by default
    img_low2 = fga.smooth(img, kernel_size, n=runs_number)  # smooth image

    plt.figure()
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title(f"original image")
    plt.subplot(222)
    plt.imshow(img_low, 'gray')
    plt.title(f"image smoothing using gaussian filter")
    plt.subplot(223)
    plt.imshow(img_low1, 'gray')
    plt.title(f"image smoothing using bilateral filter")
    plt.subplot(224)
    plt.imshow(img_low2, 'gray')
    plt.title(f"image smoothing using gradient analysis")
    plt.show()
    pass


def main():
    # canny_edge_detection()
    smoothing_algos()
    return


if __name__ == '__main__':
    main()
