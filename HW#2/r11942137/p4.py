import os
import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def wiener_filter(img, kernel, Noise_Signal_ratio):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + Noise_Signal_ratio)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

if __name__ == '__main__':
    photo_img = np.array(Image.open('./images/Photographer_degraded.tif')).astype(np.float64)
    foot_img = np.array(Image.open('./images/Football players_degraded.tif')).astype(np.float64)

    # Gaussian blur
    kernel = gaussian_kernel(3)

    # motion blur
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel_motion_blur / size

    photo_filtered_img = wiener_filter(photo_img, kernel, Noise_Signal_ratio=2)
    foot_img_filtered_img = wiener_filter(foot_img, kernel, Noise_Signal_ratio=0.5)

    plt.figure(figsize=(12, 10))
    plt.subplot(221), plt.imshow(photo_img, cmap='gray')
    plt.title("Oringinal Photographer")
    plt.subplot(222), plt.imshow(photo_filtered_img, cmap='gray')
    plt.title("Restored Photographer")
    plt.subplot(223), plt.imshow(foot_img, cmap='gray')
    plt.title("Oringinal Football")
    plt.subplot(224), plt.imshow(foot_img_filtered_img, cmap='gray')
    plt.title("Restored Football")
    plt.savefig('./images/Problem4.png')
    plt.show()