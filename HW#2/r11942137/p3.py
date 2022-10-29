import cv2
import numpy as np
import matplotlib.pyplot as plt

class ButterworthNotchFilter:
    def __init__(self, centers, size, D0=15, n=6):
        self.centers = centers
        self.size = size
        self.D0 = D0
        self.n = n
        self.filter = self.get_filter()

    def Dk(self, uk, vk):
        M, N = self.size
        M_half = M // 2
        N_half = N // 2

        D = np.empty([M, N])
        for u in range(M):
            for v in range(N):
                D[u, v] = np.sqrt((u - M_half - uk) ** 2 + (v - N_half - vk) ** 2)
        return D

    def Dnk(self, uk, vk):
        M, N = self.size
        M_half = M // 2
        N_half = N // 2

        D = np.empty([M, N])
        for u in range(M):
            for v in range(N):
                D[u, v] = np.sqrt((u - M_half + uk) ** 2 + (v - N_half + vk) ** 2)
        return D

    def get_filter(self):
        H = np.ones(self.size)
        for uk, vk in self.centers:
            d1 = self.Dk(uk, vk)
            d2 = self.Dnk(uk, vk)
            d1_power_n = d1 ** self.n
            d2_power_n = d2 ** self.n
            H *= (d1_power_n / (d1_power_n + self.D0 ** self.n)) * (d2_power_n / (d2_power_n + self.D0 ** self.n))
        return H


class IdealRecNotch:
    def __init__(self, centers, size, height, width=5):
        self.centers = centers
        self.size = size
        self.height = height
        self.width = width
        self.v_filter = self.get_v_filter()
        self.h_filter = self.get_h_filter()

    def get_v_filter(self):
        M, N = self.size
        M_half = M // 2
        N_half = N // 2
        H = np.ones(self.size)
        centers_copy = self.centers.copy()
        for u, v in self.centers:
            centers_copy.append((-u, -v))

        for uk, vk in centers_copy:
            # shift origin to left top
            uk = uk + M_half
            vk = vk + N_half
            # set four point of the filter rectangle
            ustart = uk - self.height // 2 if uk - self.height // 2 > 0 else 0
            uend = uk + self.height // 2 if uk + self.height // 2 < M else M
            vstart = vk - self.width // 2 if vk - self.width // 2 > 0 else 0
            vend = vk + self.width // 2 if vk + self.width // 2 < N else N

            H[ustart: uend, vstart: vend] = 0
        return H

    def get_h_filter(self):
        M, N = self.size
        M_half = M // 2
        N_half = N // 2
        H = np.ones(self.size)
        centers_copy = self.centers.copy()
        for u, v in self.centers:
            centers_copy.append((-u, -v))

        for uk, vk in centers_copy:
            # shift origin to left top
            uk = uk + M_half
            vk = vk + N_half
            # set four point of the filter rectangle
            ustart = uk - self.height // 2 if uk - self.height // 2 > 0 else 0
            uend = uk + self.height // 2 if uk + self.height // 2 < M else M
            vstart = vk - self.width // 2 if vk - self.width // 2 > 0 else 0
            vend = vk + self.width // 2 if vk + self.width // 2 < N else N

            H[vstart: vend, ustart: uend] = 0
        return H

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    H = np.zeros((P, Q))

    for u in range(0, P):
        for v in range(0, Q):
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)
            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0
    return H

def normalize(img_in):
    img_min = np.min(img_in)
    img_max = np.max(img_in)
    img_new = (img_in-img_min)/(img_max-img_min)*255.0
    return img_new.astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread("./images/Martian terrain.tif", cv2.IMREAD_GRAYSCALE)

    y, x = img.shape

    nsr = 2
    d = 0.5
    a = -d / x
    b = d / y
    T = 3
    u, v = np.meshgrid(np.arange(x), np.arange(y))

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    log_mag_spectrum = 1 * np.log(1 + np.abs(f_shift))

    # plt.figure(figsize=(16, 10))
    # plt.subplot(121)
    # plt.imshow(img, cmap="gray")
    # plt.title('Image')
    # plt.subplot(122)
    # plt.imshow(log_mag_spectrum, cmap='gray')
    # plt.title('Spectrum of Image')
    # plt.savefig('./images/Problem1-a.png')
    # plt.show()

    M, N = img.shape
    # (uk, vk) = filter center with repect to center(M//2, M//2)
    centers = [(-M // 2, 0)]
    ideal_rec_notch = IdealRecNotch(centers=centers, size=img.shape, height=M - 20)

    D0 = 50
    n = 5
    use_filter = np.ones((x, y))
    Dk = ((u - x // 2) ** 2 + (v - y // 2) ** 2) ** 0.5 + 1e-14
    first = 1 / (1 + ((D0 / Dk) ** n))

    f_butter = use_filter.T - first ** 2

    H1 = notch_reject_filter(img.shape, 4, 20, -40)
    H2 = notch_reject_filter(img.shape, 4, 25, 50)
    H3 = notch_reject_filter(img.shape, 4, 25, 100)
    H4 = notch_reject_filter(img.shape, 4, 2, 93)
    H5 = notch_reject_filter(img.shape, 4, 16, -88)
    H6 = notch_reject_filter(img.shape, 4, 44, -84)
    H7 = notch_reject_filter(img.shape, 4, 60, -40)
    H8 = notch_reject_filter(img.shape, 4, 55, -50)
    H9 = notch_reject_filter(img.shape, 4, 60, 46)
    H10 = notch_reject_filter(img.shape, 4, 63, 53)
    H11 = notch_reject_filter(img.shape, 4, 20, 7)

    H = H1 * H2 * H3 * H4 * H5 * H6 * H7 * H8 * H9 * H10 * H11 * ideal_rec_notch.v_filter


    # H = ideal_rec_notch.v_filter
    # H = f_butter

    # Filtering the image in freq. domain
    f_filtered_shift = H * f_shift
    f_filtered = np.fft.ifftshift(f_filtered_shift)
    spectrum_magnitude_log = np.log(1 + np.abs(f_filtered_shift))
    img_back = np.fft.ifft2(f_filtered)
    img_back = img_back.real
    img_back = cv2.equalizeHist(normalize(img_back))


    plt.figure(figsize=(12, 10))
    plt.subplot(221), plt.imshow(H, cmap='gray')
    plt.title("Ideal Notch Filter")
    plt.subplot(222), plt.imshow(spectrum_magnitude_log, cmap='gray')
    plt.title("Spectrum of Filtered Image")
    plt.subplot(223), plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.subplot(224), plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image')
    plt.savefig('./images/Problem3.png')

    plt.show()