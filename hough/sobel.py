import numpy as np
import cv2
import os
import sys
import argparse


def convolution(input_image, kernel):
    output = np.zeros_like(input_image, dtype=np.float32)

    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_input = cv2.copyMakeBorder(
        input_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE
    )

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            patch = padded_input[i : i + kernel_height, j : j + kernel_width]
            output[i, j] = np.sum(patch * kernel)

    return output


def sobel(imageName):

    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (15, 15), 3)

    image = image.astype(np.float32)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    grad_x_display = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    grad_y_display = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    magnitude_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    direction_display = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    cv2.imwrite("gradient_x.jpg", grad_x_display)
    cv2.imwrite("gradient_y.jpg", grad_y_display)
    cv2.imwrite("gradient_magnitude.jpg", magnitude_display)
    cv2.imwrite("gradient_direction.jpg", direction_display)
