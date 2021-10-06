import cv2
import numpy as np
import cannyFunctions
import sys
np.set_printoptions(threshold=sys.maxsize)

WIDTH = 600
HEIGHT = 600
rec_width = 300
rec_height = 420


def create_rectangle(img, x1, x2, y1, y2, color):
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return image


def add_noise(img):
    mean, std = cv2.meanStdDev(img)
    noise = np.random.normal(mean, std, size=img.shape)  # size = how many draws
    # apply noise and normalize values
    noisy_image = np.clip((img + noise*0.1), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def gradient_intensity(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    dx = cannyFunctions.convolve2d(img, kernel_x)
    dy = cannyFunctions.convolve2d(img, kernel_y)
    g = np.hypot(dx, dy)
    d = np.arctan2(dy, dx)
    return (g, d)


def canny(image, kernel, low_thresh=50, high_thresh=90):
    smoothed7 = cannyFunctions.convolve2d(image, kernel)
    smoothed7x2 = cannyFunctions.convolve2d(smoothed7, kernel)
    gradient, d = gradient_intensity(np.asarray(smoothed7x2, dtype="int32"))
    suppressed = cannyFunctions.suppression(gradient, d)
    th, weak = cannyFunctions.threshold(suppressed, low_thresh, high_thresh)
    tracked = cannyFunctions.tracking(th, weak)
    return tracked


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height))) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def get_rect_points(acc, thetas, rhos):
    points = list()
    minimum = np.argmin(acc) - 1
    # get maximum values to draw needed lines
    for i in range(28): # 50 30
        idx = np.argmax(acc)
        rho = rhos[idx // acc.shape[1]]
        theta = thetas[idx % acc.shape[1]]
        acc[idx // acc.shape[1], idx % acc.shape[1]] = minimum
        points.append((rho, theta))
    return points


def convert_points(points, img):
    lines = list()
    for rho, theta in points:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # calculate x and y points according to rectangle
        x1 = int(x0 + img.shape[0] * (-b))
        y1 = int(y0 + img.shape[1] * a)
        x2 = int(x0 - img.shape[0] * (-b))
        y2 = int(y0 - img.shape[1] * a)
        if x1 != x2 and y1 != y2:
            lines.append([(x2, y2), (x1, y1)])
    return lines


# returns tuple (x, y) or None if there is no intersection
def intersection(a1, a2, b1, b2):
    if type(a1) != type(a2) != type(b1) != type(b2) != tuple:
        return
    if len(a1) != len(a2) != len(b1) != len(b2) != 2:
        return
    a_x1, a_y1 = a1
    a_x2, a_y2 = a2
    b_x1, b_y1 = b1
    b_x2, b_y2 = b2
    d = (b_y2 - b_y1) * (a_x2 - a_x1) - (b_x2 - b_x1) * (a_y2 - a_y1)
    if d:
        u_a = ((b_x2 - b_x1) * (a_y1 - b_y1) - (b_y2 - b_y1) * (a_x1 - b_x1)) / d
        u_b = ((a_x2 - a_x1) * (a_y1 - b_y1) - (a_y2 - a_y1) * (a_x1 - b_x1)) / d
    else:
        return
    if not (0 <= u_a <= 1 and 0 <= u_b <= 1):
        return
    x = a_x1 + u_a * (a_x2 - a_x1)
    y = a_y1 + u_a * (a_y2 - a_y1)

    return x, y
