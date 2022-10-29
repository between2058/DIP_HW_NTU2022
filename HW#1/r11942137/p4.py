import numpy as np
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(l, sigma):
    ax = np.linspace(-(l-1)/2., (l-1)/2., l)
    gauss = np.exp(-0.5*np.square(ax)/np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def conv(image, kernel, stride=1):
    if kernel.shape[0] % 2 == 1:
        padding = kernel.shape[0] // 2
    else:
        padding = kernel.shape[0] // 2 + 1

    target_shape = image.shape
    kernel_shape = kernel.shape
    result = np.zeros_like(image)


    padding_left = np.zeros((image.shape[0], padding), np.float32)
    padding_right = padding_left.copy()
    # padding_right.fill(255)
    print(image.shape, padding_right.shape, padding_left.shape)
    image = np.concatenate((padding_left, image, padding_right), 1)

    padding_top = np.zeros((padding, image.shape[1]), np.float32)
    padding_bottom = padding_top.copy()
    # padding_bottom.fill(255)
    image = np.concatenate((padding_top, image, padding_bottom), 0)

    for i in range(0, target_shape[0], stride):
        for j in range(0,target_shape[1], stride):
            window = image[i:i + kernel_shape[0], j:j + kernel_shape[1]]
            val = window * kernel
            result[i,j] = val.sum()

    return result

if __name__ == '__main__':
    img = cv2.imread("./images/image_4.tif", 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fres = np.log(np.abs(fshift))

    # f = 1*np.log(1+np.abs(f))

    kernel = gaussian_kernel(l=300, sigma=64.)
    lowpass = conv(img, kernel, stride=1)
    # for i in range(2):
    #     print("times:", i+1)
    #     lowpass = shading_correction(lowpass, kernel, stride=1, padding="zero")
    # result = img/lowpass

    l = np.fft.fft2(lowpass)
    lshift = np.fft.fftshift(l)
    lres =  np.log(np.abs(lshift))

    res = np.fft.ifftshift(np.divide(fshift,lshift))
    res = np.fft.ifft2(res)
    res = np.abs(res)

    result = np.divide(img, lowpass)


    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    ax1.imshow(img, cmap='gray')
    ax2.imshow(kernel, cmap='gray')
    ax3.imshow(lowpass, cmap='gray')
    ax4.imshow(result, cmap='gray')
    ax5.imshow(res, cmap='gray')
    plt.savefig("./images/shading_correction.png")
    plt.show()
