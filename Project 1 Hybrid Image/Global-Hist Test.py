import numpy as np
import math
import cv2


def cross_correlation_2d(kernel, img):
    m, n = np.shape(kernel)
    row, col = np.shape(img)
    pad_num = m // 2
    ans = np.zeros((row-2*pad_num, col-2*pad_num))
    for i in range(pad_num, row-pad_num):
        for j in range(pad_num, col-pad_num):
            cal_matrix = img[i-pad_num:i+pad_num+1, j-pad_num:j+pad_num+1]
            dot_ans = np.multiply(kernel, cal_matrix)
            ans[i-pad_num][j-pad_num] = dot_ans.sum()
    return ans


def convolve_2d(kernel, img):
    m, n = np.shape(kernel)
    convolve_kernel = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            convolve_kernel[i][j] = kernel[m-i-1][n-j-1]
    return cross_correlation_2d(convolve_kernel, img)


def gaussian_blur_kernel_2d(n, sigma):  # n kernel size
    if n % 2 == 0:
        raise Exception("Please enter a odd number for the gaussian filter!")
    else:
        temp_kernel = np.zeros((n, n))
        center = n // 2 + 1
        for i in range(n):
            for j in range(n):
                temp_kernel[i][j] = math.exp(-((i - center + 1) ** 2 + (j - center + 1) ** 2) / (2 * sigma ** 2))
        kernel = temp_kernel / temp_kernel.sum()
    return kernel


def low_pass(img1, n, sigma):
    img = cv2.imread(img1)
    b, g, r = cv2.split(img)
    pad_num = n // 2
    kernel = gaussian_blur_kernel_2d(n, sigma)
    padded_b = np.pad(b, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
    padded_g = np.pad(g, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
    padded_r = np.pad(r, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
    b = convolve_2d(kernel, padded_b)
    g = convolve_2d(kernel, padded_g)
    r = convolve_2d(kernel, padded_r)
    low_pass_img = cv2.merge([b, g, r])
    return low_pass_img.astype(np.uint8)


def high_pass(img2, n, sigma):
    blur_img = low_pass(img2, n, sigma)
    img = cv2.imread(img2)
    high_pass_img = img - blur_img + 128
    return high_pass_img


def hybrid(img1, img2, n, sigma, alpha):
    low_pass_img = low_pass(img1, n, sigma)
    high_pass_img = high_pass(img2, n, sigma)
    hybrid_img = alpha * low_pass_img + (1 - alpha) * high_pass_img
    hybrid_img = hybrid_img.astype(np.uint8)
    b, g, r = cv2.split(hybrid_img)
    global_hist_b = cv2.equalizeHist(b)
    global_hist_g = cv2.equalizeHist(g)
    global_hist_r = cv2.equalizeHist(r)
    return cv2.merge([global_hist_b, global_hist_g, global_hist_r])


# wheel_test1
# a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# wheel_test1 = cross_correlation_2d(a, b)
# print(wheel_test1)

# wheel_test2
# wheel_test2 = convolve_2d(a, b)
# print(wheel_test2)

# wheel_test3
# wheel_test3 = gaussian_blur_kernel_2d(5, 1)
# print(wheel_test3)

# test = cv2.getGaussianKernel(5, 10)
# print(test)
# print(np.transpose(test))
# result = np.dot(test, np.transpose(test))
# print(result)

# wheel_test4
# test4 = low_pass('dog.jpg', 15, 10)
# original4 = cv2.imread('dog.jpg')
# cv2.imshow('Original Image', original4)
# cv2.imshow('Low_Pass Image', test4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# wheel_test5
# test5 = high_pass('cat.jpg', 15, 10)
# original5 = cv2.imread('cat.jpg')
# cv2.imshow('Original Image', original5)
# cv2.imshow('High_Pass Image', test5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# wheel_test6
test6 = hybrid('dog.jpg', 'cat.jpg', 15, 10, 0.6)
left_img = cv2.imread('dog.jpg')
right_img = cv2.imread('cat.jpg')
cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.imshow('Hybrid Image', test6)
cv2.waitKey(0)
cv2.destroyAllWindows()