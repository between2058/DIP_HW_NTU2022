import cv2
import numpy as np
import matplotlib.pyplot as plt

def HE(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 255])
    pdf = hist/img.size
    cdf = pdf.cumsum()
    equ_value = np.around(cdf * 255).astype('uint8')
    result = equ_value[img]
    return result

def GC(img, gamma):
    intensity = img/float(np.max(img))
    result = np.power(intensity, gamma)
    return result

if __name__ == '__main__':
    img = cv2.imread("./images/image_2.png")
    HE = HE(img)

    plt.subplot(2,2,1)
    plt.title("Original image")
    plt.imshow(img)

    plt.subplot(2,2,2)
    plt.title("Original histogram")
    plt.hist(img.ravel(), bins=256, range=(0, 255), color='b')

    plt.subplot(2,2,3)
    plt.title("Equalized image")
    plt.imshow(HE)

    plt.subplot(2,2,4)
    plt.title("Equalized histogram")
    plt.hist(HE.ravel(), bins=256, range=(0, 255), color='b')
    plt.show()
#     ==================================HE end========================================
#     ==================================GC start======================================
    plt.subplot(2,5,1)
    plt.title("Gamma = 0.1")
    plt.imshow(GC(img,0.1))


    plt.subplot(2,5,2)
    plt.title("Gamma = 0.2")
    plt.imshow(GC(img,0.2))


    plt.subplot(2,5,3)
    plt.title("Gamma = 0.4")
    plt.imshow(GC(img,0.4))


    plt.subplot(2,5,4)
    plt.title("Gamma = 0.67")
    plt.imshow(GC(img,0.67))


    plt.subplot(2,5,5)
    plt.title("Gamma = 1")
    plt.imshow(GC(img,1))


    plt.subplot(2,5,6)
    plt.title("Gamma = 1.5")
    plt.imshow(GC(img,1.5))


    plt.subplot(2,5,7)
    plt.title("Gamma = 2.5")
    plt.imshow(GC(img,2.5))


    plt.subplot(2,5,8)
    plt.title("Gamma = 5")
    plt.imshow(GC(img,5))


    plt.subplot(2,5,9)
    plt.title("Gamma = 10")
    plt.imshow(GC(img,10))


    plt.subplot(2,5,10)
    plt.title("Gamma = 25")
    plt.imshow(GC(img,25))


    plt.show()



