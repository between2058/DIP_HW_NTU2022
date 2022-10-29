import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def shift_mask(h,w):
    shift = np.empty([h,w])
    for i in range(h):
        for j in range(w):
            shift[i,j] = (-1) ** (i+j)
    return shift

def HE(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 255])
    pdf = hist/img.size
    cdf = pdf.cumsum()
    equ_value = np.around(cdf * 255).astype('uint8')
    result = equ_value[img]
    return result


if __name__ == '__main__':
    img = cv2.imread("./images/Einstein.tif", 0)

    h, w = img.shape
    P = h + 3 - 1
    Q = w + 3 - 1

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # create F(u,v)
    padded_img = np.zeros([P, Q])
    padded_img[:h, :w] = img
    shift = shift_mask(h, w)
    padded_shifted_img = np.zeros([P, Q])
    padded_shifted_img[:h, :w] = img * shift
    f_padded_shifted_img = np.fft.fft2(padded_shifted_img)
    log_f_padded_shifted_img = 1 * np.log(1 + np.abs(f_padded_shifted_img))

    # create H(u,v)
    kernel_size = kernel.shape[0]
    shifted_sobel = np.zeros([P, Q])
    shifted_sobel[:kernel_size, :kernel_size] = kernel * shift_mask(kernel_size, kernel_size)
    f_shifted_sobel = np.fft.fft2(shifted_sobel)
    log_f_shifted_sobel = 1 * np.log(1 + np.abs(f_shifted_sobel))

    # G(u,v) = F(u,v)*H(u,v)
    f_filtered_img = f_padded_shifted_img * f_shifted_sobel
    log_f_filtered_img = 1 * np.log(1 + np.abs(f_filtered_img))

    # Get back to spatial domain
    img_filtered = np.fft.ifft2(f_filtered_img).real * shift_mask(P, Q)
    img_eq = img_filtered[:h, :w]

    plt.figure(figsize=(16, 10))

    plt.subplot(131)
    plt.imshow(log_f_padded_shifted_img, cmap='gray')
    plt.title('Spectrum of padded and shifted image')
    plt.subplot(132)
    plt.imshow(log_f_shifted_sobel, cmap='gray')
    plt.title('Spectrum of padded and shifted Sharpen filter')
    plt.subplot(133)
    plt.imshow(log_f_filtered_img, cmap='gray')
    plt.title('Spectrum of filtered image')
    plt.savefig('./images/Problem2-1.png')
    plt.show()

    img_eq = np.clip(img_eq, 0, 255)
    img_eq = img_eq/img_eq.max()
    img_eq *= img.max()
    img_eq[img_eq - img_eq.mean() < 0.15] = 0
    cv2.imshow("edge_Einstein", img_eq)

    for i in range(10):
        img_eq += img
        img_eq -= img_eq.min()
        img_eq = img_eq / img_eq.max()
        img_eq *= img.max()


    cv2.imshow("Einstein",img)
    cv2.imshow("Einstein_eq", np.array(img_eq, dtype=np.uint8))
    cv2.imwrite("./images/filtered_Einstein.png", np.array(img_eq, dtype=np.uint8))

    # =============================================================================================

    img2 = cv2.imread("./images/phobos.tif", 0)
    img_he = HE(img2)
    cv2.imshow("he_phobos", img_he)

    h, w = img_he.shape
    P = h + 3 - 1
    Q = w + 3 - 1

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # create F(u,v)
    padded_img = np.zeros([P, Q])
    padded_img[:h, :w] = img_he
    shift = shift_mask(h, w)
    padded_shifted_img = np.zeros([P, Q])
    padded_shifted_img[:h, :w] = img_he * shift
    f_padded_shifted_img = np.fft.fft2(padded_shifted_img)
    log_f_padded_shifted_img = 1 * np.log(1 + np.abs(f_padded_shifted_img))

    # create H(u,v)
    kernel_size = kernel.shape[0]
    shifted_sobel = np.zeros([P, Q])
    shifted_sobel[:kernel_size, :kernel_size] = kernel * shift_mask(kernel_size, kernel_size)
    f_shifted_sobel = np.fft.fft2(shifted_sobel)
    log_f_shifted_sobel = 1 * np.log(1 + np.abs(f_shifted_sobel))

    # G(u,v) = F(u,v)*H(u,v)
    f_filtered_img = f_padded_shifted_img * f_shifted_sobel
    log_f_filtered_img = 1 * np.log(1 + np.abs(f_filtered_img))

    # Get back to spatial domain
    img_filtered = np.fft.ifft2(f_filtered_img).real * shift_mask(P, Q)
    img_eq2 = img_filtered[:h, :w]

    plt.figure(figsize=(16, 10))

    plt.subplot(131)
    plt.imshow(log_f_padded_shifted_img, cmap='gray')
    plt.title('Spectrum of padded and shifted image')
    plt.subplot(132)
    plt.imshow(log_f_shifted_sobel, cmap='gray')
    plt.title('Spectrum of padded and shifted Sharpen filter')
    plt.subplot(133)
    plt.imshow(log_f_filtered_img, cmap='gray')
    plt.title('Spectrum of filtered image')
    plt.savefig('./images/Problem2-2.png')
    plt.show()



    img_eq2 = np.clip(img_eq2, 0, 255)
    img_eq2 = img_eq2 / img_eq2.max()
    img_eq2 *= img.max()
    img_eq2[img_eq2 - img_eq2.mean() < 0.15] = 0
    cv2.imshow("conv_phobos", img_eq2)
    for i in range(10):
        img_eq2 += img_he
        img_eq2 -= img_eq2.min()
        img_eq2 = img_eq2 / img_eq2.max()
        img_eq2 *= img_he.max()

    cv2.imshow("phobos", img2)
    cv2.imshow("phobos_eq", np.array(img_eq2, dtype=np.uint8))
    cv2.imwrite("./images/filtered_phobos.png", np.array(img_eq2, dtype=np.uint8))

    cv2.waitKey(0)


