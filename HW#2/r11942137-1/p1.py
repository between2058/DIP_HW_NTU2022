import cv2
import numpy as np
import matplotlib.pyplot as plt

def shift_mask(h,w):
    shift = np.empty([h,w])
    for i in range(h):
        for j in range(w):
            shift[i,j] = (-1) ** (i+j)
    return shift

def zero_padding(img, kernel_size):
    row, col = img.shape[:2]
    # odd kernel size
    if kernel_size % 2 == 1:
        num_to_pad = kernel_size // 2 * 2
        offset = num_to_pad // 2
    # even kernel size
    else:
        num_to_pad = kernel_size - 1
        offset = kernel_size // 2 - 1

    img_padded = np.zeros([row + num_to_pad, col + num_to_pad])
    img_padded[offset: offset + row, offset: offset + col] = img

    return img_padded

def conv2d(img, kernel):
    kernel_size = kernel.shape[0]
    img_padded = zero_padding(img, kernel_size)
    src_row, src_col = img.shape
    img_filtered = np.empty((src_row, src_col))
    kernel = np.flipud(np.fliplr(kernel))
    for row in range(src_row):
        for col in range(src_col):
            img_filtered[row, col] = np.sum(
                np.multiply(kernel, img_padded[row:row + kernel_size, col: col + kernel_size]))

    return img_filtered

if __name__ == '__main__':

    img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    log_mag_spectrum = 1 * np.log(1+ np.abs(f_shift))

    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title('Image')
    plt.subplot(122)
    plt.imshow(log_mag_spectrum, cmap='gray')
    plt.title('Spectrum of Image')
    plt.savefig('./images/Problem1-a.png')
    plt.show()



    sobel = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
    # sobel = np.array([[-1,-2,-1],
    #                 [0,0,0],
    #                 [1,2,1]])
    sobel_odd_sym = np.zeros([4,4])
    sobel_odd_sym[1:, 1:] = sobel
    print("Original sobel kernel:\n", sobel)
    print("After enforcing odd symmetry:\n", sobel_odd_sym.astype(np.int32))

    h,w = img.shape
    P = h + 3 - 1
    Q = w + 3 - 1

    # create F(u,v)
    padded_img = np.zeros([P,Q])
    padded_img[:h, :w] = img
    shift = shift_mask(h, w)
    padded_shifted_img = np.zeros([P,Q])
    padded_shifted_img[:h, :w] = img * shift
    f_padded_shifted_img = np.fft.fft2(padded_shifted_img)
    log_f_padded_shifted_img = 1 * np.log(1+ np.abs(f_padded_shifted_img))

    # create H(u,v)
    kernel_size = sobel_odd_sym.shape[0]
    shifted_sobel = np.zeros([P,Q])
    shifted_sobel[:kernel_size, :kernel_size] = sobel_odd_sym * shift_mask(kernel_size,kernel_size)
    f_shifted_sobel = np.fft.fft2(shifted_sobel)
    log_f_shifted_sobel = 1 * np.log(1+ np.abs(f_shifted_sobel))

    # G(u,v) = F(u,v)*H(u,v)
    f_filtered_img = f_padded_shifted_img * f_shifted_sobel
    log_f_filtered_img = 1 * np.log(1 + np.abs(f_filtered_img))

    # Get back to spatial domain
    img_filtered = np.fft.ifft2(f_filtered_img).real * shift_mask(P, Q)
    img_filtered = img_filtered[:h, :w]

    # Compare with filtering in spatial domain
    img_spatial_filtered = conv2d(img, sobel)
    img_spatial_filtered_magnitude = np.abs(img_spatial_filtered)




    plt.figure(figsize=(16, 10))

    plt.subplot(221)
    plt.imshow(log_f_padded_shifted_img, cmap='gray')
    plt.title('Spectrum of padded and shifted image')
    plt.subplot(222)
    plt.imshow(log_f_shifted_sobel, cmap='gray')
    plt.title('Spectrum of padded and shifted Sobel Kernel')
    plt.subplot(223)
    plt.imshow(log_f_filtered_img, cmap='gray')
    plt.title('Spectrum of filtered image')
    plt.subplot(224)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Filtered image')
    plt.savefig('./images/Problem1-c.png')
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Fitered image using frequency domain')
    plt.subplot(122)
    plt.imshow(img_spatial_filtered, cmap='gray')
    plt.title('Fitered image using spatial domain')
    plt.savefig('./images/Problem1-d.png')
    plt.show()

    kernel_size = sobel.shape[0]
    shifted_sobel = np.zeros([P, Q])
    shifted_sobel[:kernel_size, :kernel_size] = sobel * shift_mask(kernel_size, kernel_size)
    f_shifted_sobel = np.fft.fft2(shifted_sobel)
    f_shifted_sobel_mag = np.abs(f_shifted_sobel)

    f_filtered_img = f_padded_shifted_img * f_shifted_sobel_mag

    img_filtered = np.fft.ifft2(f_filtered_img).real * shift_mask(P, Q)
    img_filtered = img_filtered[:h, :w]


    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Fitered image without enforce odd symmetry using frequency domain')
    plt.subplot(122)
    plt.imshow(img_spatial_filtered, cmap='gray')
    plt.title('Fitered image using spatial domain')
    plt.savefig('./images/Problem1-e.png')
    plt.show()


