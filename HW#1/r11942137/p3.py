import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def nearest_interpolation(image, scale, degree):

    rads = math.radians(degree)

    image_width = image.shape[0]
    image_height = image.shape[1]

    resized_image_width = math.floor(image_width * scale)
    resized_image_height = math.floor(image_height * scale)

    midx, midy = (resized_image_width // 2 + 40, resized_image_height // 2 + 20)

    image_output = np.zeros((resized_image_width, resized_image_height), dtype=np.uint8)
    image_output.fill(255)

    for x in range(resized_image_width):

        x_ori = (x / resized_image_width) * image_width
        x_interp = x_ori - np.floor(x_ori)

        # find the nearest neighbour of current x
        if x_interp < 0.5:
            x_int = int(np.floor(x_ori))
        else:
            x_int = int(np.ceil(x_ori))
            # if x_int is out of bound force it back inbound
            if x_int >= image_width:
                x_int = int(np.floor(x_ori))

        for y in range(resized_image_height):
            y_ori = (y / resized_image_height) * image_height

            y_interp = y_ori - np.floor(y_ori)

            # find the nearest neighbour of current y
            if y_interp < 0.5:
                y_int = int(np.floor(y_ori))
            else:
                y_int = int(np.ceil(y_ori))
                # if y_int is out of bound force it back inbound
                if y_int >= image_height:
                    y_int = int(np.floor(y_ori))

            X_int = (x_int - midx) * math.cos(rads) + (y_int - midy) * math.sin(rads)
            Y_int = -(x_int - midx) * math.sin(rads) + (y_int - midy) * math.cos(rads)
            X_int = round(X_int) + midx
            Y_int = round(Y_int) + midy
            if (X_int >= 0 and Y_int >= 0 and X_int < image.shape[0] and Y_int < image.shape[1]):
                image_output[x, y] = image[X_int, Y_int]

    return image_output

def bilinear(image, scale, degree):

    rads = math.radians(degree)

    image_width = image.shape[0]
    image_height = image.shape[1]

    resized_image_width = math.floor(image_width * scale)
    resized_image_height = math.floor(image_height * scale)

    midx, midy = (resized_image_width // 2 + 40, resized_image_height // 2 + 20)

    B = np.array([[0, 1], [1, 1]])
    B_inv = np.linalg.inv(B)  # since B is symmetric matrix B_inv = B_inv_T

    image_output = np.zeros((resized_image_width, resized_image_height), dtype=np.uint8)
    image_output.fill(255)


    # pad row
    bottom_row = image[-1, :]
    image_row_padding = np.vstack((image, bottom_row))

    # pad column
    rightmost_column = image_row_padding[:, -1]
    image_padding = np.c_[image_row_padding, rightmost_column]

    F = np.zeros((image_width, image_height, 2, 2))

    for x in range(image_width):
        for y in range(image_height):
            f = np.array([[image_padding[x][y], image_padding[x][y + 1]],
                          [image_padding[x + 1][y], image_padding[x + 1][y + 1]]])
            F[x][y] = f

    # interpolate image pixel by pixel
    for x in range(resized_image_width):

        x_ori = (x / resized_image_width) * image_width
        x_interp = x_ori - np.floor(x_ori)
        x_int = int(np.floor(x_ori))

        for y in range(resized_image_height):

            y_ori = (y / resized_image_height) * image_height
            y_interp = y_ori - np.floor(y_ori)
            y_int = int(np.floor(y_ori))

            X_int = (x_int - midx) * math.cos(rads) + (y_int - midy) * math.sin(rads)
            Y_int = -(x_int - midx) * math.sin(rads) + (y_int - midy) * math.cos(rads)
            X_int = round(X_int) + midx
            Y_int = round(Y_int) + midy
            if (X_int >= 0 and Y_int >= 0 and X_int < image.shape[0] and Y_int < image.shape[1]):


                if x_interp == 0.0 and y_interp == 0.0:
                    image_output[x][y] = image[int(x_ori)][int(y_ori)]
                else:
                    # interpolate value in x direction (row vector)
                    X = np.expand_dims(np.array([x_interp ** 1, x_interp ** 0]), axis=0)
                    # interpolate value in y direction (column vector)
                    Y = np.expand_dims(np.array([y_interp ** 1, y_interp ** 0]), axis=1)


                    F_interp = F[X_int][Y_int]


                    # bilinear interpolation
                    interpolated_value = X.dot(B_inv).dot(F_interp).dot(B_inv).dot(Y)
                    if interpolated_value < 0:
                        interpolated_value = 0
                    elif interpolated_value > 255:
                        interpolated_value = 255

                    image_output[x][y] = interpolated_value



    return image_output


def bicubic(image, scale, degree):
    rads = math.radians(degree)

    image_width = image.shape[0]
    image_height = image.shape[1]

    resized_image_width = math.floor(image_width * scale)
    resized_image_height = math.floor(image_height * scale)

    midx, midy = (resized_image_width // 2 + 40, resized_image_height // 2 + 20)

    B = np.array([[-1, 1, -1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [8, 4, 2, 1]])
    B_inv = np.linalg.inv(B)
    B_inv_T = B_inv.T

    image_output = np.zeros((resized_image_width, resized_image_height), dtype=np.uint8)
    image_output.fill(255)

    # pad row
    top_row = image[0, :]
    bottom_row = image[-1, :]
    image_row_padding = np.vstack((top_row, image))
    image_row_padding = np.vstack((image_row_padding, bottom_row))
    image_row_padding = np.vstack((image_row_padding, bottom_row))

    # pad column
    leftmost_column = image_row_padding[:, 0]
    rightmost_column = image_row_padding[:, -1]
    image_padding = np.c_[leftmost_column, image_row_padding, rightmost_column, rightmost_column]

    F = np.zeros((image_width, image_height, 4, 4))

    for x in range(image_width):
        x_padding = x + 1
        for y in range(image_height):
            y_padding = y + 1

            # formalize f matrix
            f = np.array([[image_padding[x_padding - 1][y_padding - 1], image_padding[x_padding - 1][y_padding],
                           image_padding[x_padding - 1][y_padding + 1],
                           image_padding[x_padding - 1][y_padding + 2]],
                          [image_padding[x_padding][y_padding - 1], image_padding[x_padding][y_padding],
                           image_padding[x_padding][y_padding + 1], image_padding[x_padding][y_padding + 2]],
                          [image_padding[x_padding + 1][y_padding - 1], image_padding[x_padding + 1][y_padding],
                           image_padding[x_padding + 1][y_padding + 1],
                           image_padding[x_padding + 1][y_padding + 2]],
                          [image_padding[x_padding + 2][y_padding - 1], image_padding[x_padding + 2][y_padding],
                           image_padding[x_padding + 2][y_padding + 1],
                           image_padding[x_padding + 2][y_padding + 2]]])
            F[x][y] = f

    # interpolate image pixel by pixel
    for x in range(resized_image_width):

        x_ori = (x / resized_image_width) * image_width
        x_interp = x_ori - np.floor(x_ori)
        x_int = int(np.floor(x_ori))

        for y in range(resized_image_height):

            y_ori = (y / resized_image_height) * image_height
            y_interp = y_ori - np.floor(y_ori)
            y_int = int(np.floor(y_ori))

            X_int = (x_int - midx) * math.cos(rads) + (y_int - midy) * math.sin(rads)
            Y_int = -(x_int - midx) * math.sin(rads) + (y_int - midy) * math.cos(rads)
            X_int = round(X_int) + midx
            Y_int = round(Y_int) + midy
            if (X_int >= 0 and Y_int >= 0 and X_int < image.shape[0] and Y_int < image.shape[1]):

                # 直接copy，不用interpolation
                if x_interp == 0.0 and y_interp == 0.0:
                    image_output[x][y] = image[int(x_ori)][int(y_ori)]

                # 分別對 x、y
                else:
                    # interpolate value in x direction (row vector)
                    X = np.expand_dims(np.array([x_interp ** 3, x_interp ** 2, x_interp ** 1, x_interp ** 0]), axis=0)
                    # interpolate value in y direction (column vector)
                    Y = np.expand_dims(np.array([y_interp ** 3, y_interp ** 2, y_interp ** 1, y_interp ** 0]), axis=1)

                    F_interp = F[X_int][Y_int]

                    #bicubic interpolation
                    interpolated_value = X.dot(B_inv).dot(F_interp).dot(B_inv_T).dot(Y)

                    if interpolated_value < 0:
                        interpolated_value = 0
                    elif interpolated_value > 255:
                        interpolated_value = 255

                    image_output[x][y] = interpolated_value

    return image_output

if __name__ == '__main__':
    img = cv2.imread("./images/T.png",0)
    nni = nearest_interpolation(img, 0.7, -45)
    bl = bilinear(img, 0.7, -45)
    bc = bicubic(img, 0.7, -45)


    cv2.imshow("original",img)
    cv2.imshow("nearest",nni)
    cv2.imshow("bilinear", bl)
    cv2.imshow("bicubic", bc)
    cv2.imwrite("./images/nearest_interpolation.png",nni)
    cv2.imwrite("./images/bilinear_interpolation.png", bl)
    cv2.imwrite("./images/bicubic_interpolation.png", bc)
    cv2.waitKey(0)

    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
    # ax1.imshow(img)
    # ax2.imshow(nni)
    # plt.show()