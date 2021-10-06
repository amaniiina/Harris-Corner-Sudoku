import cv2
import numpy as np
import houghFunctions
from matplotlib import pyplot as plt


def convolve2d(image, kernel):
    output = np.zeros(image.shape, image.dtype)
    ker_rows = kernel.shape[0]
    # set padding size depending on kernel size
    pad_size = kernel.shape[0] // 2
    if pad_size == 0:
        pad_size = 1
    # Add zero padding to the image
    padded = np.zeros((image.shape[0] + 2 * pad_size, image.shape[1] + 2 * pad_size))
    padded[pad_size:-pad_size, pad_size:-pad_size] = image
    # for every pixel multiply the kernel and the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum((kernel * padded[i: i+ker_rows, j: j+ker_rows]))
    return output


def gradient(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    dx = convolve2d(img, kernel_x)
    dy = convolve2d(img, kernel_y)
    return dx, dy


def harris_corners(img, k, offset, gaussian):
    corners = []
    y_range = img.shape[0]
    x_range = img.shape[1]
    dy, dx = np.gradient(img)
    ixx = dx ** 2
    ixx = convolve2d(ixx, gaussian)
    ixy = dy * dx
    ixy = convolve2d(ixy, gaussian)
    iyy = dy ** 2
    iyy = convolve2d(iyy, gaussian)
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            # Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            window_ixx = ixx[start_y: end_y, start_x: end_x]
            window_ixy = ixy[start_y: end_y, start_x: end_x]
            window_iyy = iyy[start_y: end_y, start_x: end_x]
            # sum intensities of derivatives
            Sxx = window_ixx.sum()
            Sxy = window_ixy.sum()
            Syy = window_iyy.sum()
            # calc det and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            # calculate r
            r = det - k * (trace ** 2)
            # r > 0 -> corner, r < 0 -> edge, r == 0 -> flat
            if r > 0:
                corners.append([x, y, r])
    return corners


def dilate(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                if i > 0 and image[i-1][j] == 0:
                    image[i-1][j] = 2
                if j > 0 and image[i][j-1] == 0:
                    image[i][j-1] = 2
                if i+1 < image.shape[0] and image[i+1][j] == 0:
                    image[i+1][j] = 2
                if j+1 < image.shape[1] and image[i][j+1] == 0:
                    image[i][j+1] = 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 2:
                image[i][j] = 255
    return image


def hough_img(img):
    # perform hough transform
    acc, thetas, rhos = houghFunctions.hough_line(img[:img.shape[0] - 20, :])
    # get polar points and convert them to cartesian
    points = houghFunctions.get_rect_points(acc, thetas, rhos)
    lines = houghFunctions.convert_points(points, np.copy(img))
    # calculate intersections and append intersecting lines to list final_lines
    final_lines = list()
    for index, line in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index != index2:
                point = houghFunctions.intersection(line[0], line[1], line2[0], line2[1])
                if point and img.shape[1] > int(point[0]) > 0 and img.shape[0] > int(point[1]) > 0:
                    final_lines.append(line)
                    final_lines.append(line2)

    # draw intersecting lines of outer grid on new image with black background
    new_img = np.zeros_like(img, np.uint8)
    for line in final_lines:
        cv2.line(new_img, line[0], line[1], 255, 2)
    return new_img


def main():
    # read image in grayscale and convert it to float32
    image = np.array(cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE), np.float32)
    # create 3 channel copy of image for result
    output = cv2.cvtColor(np.uint8(image).copy(), cv2.COLOR_GRAY2RGB)
    gaussian7x7 = np.array([[1, 6, 15, 20, 15, 6, 1],
                            [6, 36, 90, 120, 90, 36, 6],
                            [15, 90, 225, 300, 225, 90, 15],
                            [20, 120, 300, 400, 300, 120, 20],
                            [15, 90, 225, 300, 225, 90, 15],
                            [6, 36, 90, 120, 90, 36, 6],
                            [1, 6, 15, 20, 15, 6, 1]]) / 4096
    # threshold image
    thresh_img = 255 - image
    thresh_img[thresh_img > 210] = 255
    thresh_img[thresh_img <= 210] = 0
    # perform canny algorithm to get edges
    thresh_img = houghFunctions.canny(thresh_img, gaussian7x7, 30, 45)
    # perform hough transform on canny result to get clearer edges
    hough_image = hough_img(thresh_img)

    # dilate image and blur it
    for i in range(2):
        dilate(hough_image)
    for i in range(4): # 4 great try 3
        hough_image = convolve2d(hough_image, gaussian7x7)

    # Value of Harris corner constant between 0.04 - 0.06
    k = 0.04
    window_size = 3
    offset = int(window_size / 2)
    # use harris corner detection and mark the returned pixels
    corner_list = harris_corners(hough_image, k, offset, gaussian7x7)
    for (x, y, r) in corner_list:
        # to not get extra corners from the edges in the final image
        if 40<y<output.shape[1]-10 and 20<x<output.shape[0]-13:
            output[int(y), int(x)] = (255, 0, 0)

    # show result
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(output)
    plt.title('Corners'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
