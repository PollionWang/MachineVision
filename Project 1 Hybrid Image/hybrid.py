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
    clahe = cv2.createCLAHE(1.5, (1, 1))
    local_hist_b = clahe.apply(b)
    local_hist_g = clahe.apply(g)
    local_hist_r = clahe.apply(r)
    return cv2.merge([local_hist_b, local_hist_g, local_hist_r])


hybrid_ans = hybrid('left.jpg', 'right.jpg', 15, 10, 0.6)
cv2.imwrite('hybrid.jpg',hybrid_ans)
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')
cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.imshow('Hybrid Image', hybrid_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()