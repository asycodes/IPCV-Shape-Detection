import cv2
import numpy as np


def hough_circle_with_orientation(imageName, min_radius, max_radius, threshold_hough):

    gradient_magnitude = cv2.imread("gradient_magnitude.jpg", cv2.IMREAD_GRAYSCALE)
    gradient_direction = cv2.imread("gradient_direction.jpg", cv2.IMREAD_GRAYSCALE)

    # CONVERT TO RADIANSSSSSS
    gradient_direction = gradient_direction.astype(np.float32)
    gradient_direction = (gradient_direction / 255) * 2 * np.pi  # Scale to [0, 2π]

    threshold_value = 70
    _, thresholded = cv2.threshold(
        gradient_magnitude, threshold_value, 255, cv2.THRESH_BINARY
    )
    cv2.imwrite("thresholded_gradient_magnitude.png", thresholded)
    height, width = thresholded.shape
    hough_space = np.zeros((height, width, max_radius), dtype=np.uint64)

    edge_points = np.argwhere(thresholded > 0)

    for y, x in edge_points:
        theta = gradient_direction[y, x]
        for r in range(min_radius, max_radius):
            a1 = int(x - r * np.cos(theta))
            b1 = int(y - r * np.sin(theta))

            if 0 <= a1 < width and 0 <= b1 < height:
                hough_space[b1, a1, r] += 1

            a2 = int(x + r * np.cos(theta))
            b2 = int(y + r * np.sin(theta))

            if 0 <= a2 < width and 0 <= b2 < height:
                hough_space[b2, a2, r] += 1

    detected_circles = []
    for r in range(min_radius, max_radius):
        hough_slice = hough_space[:, :, r]
        peaks = np.argwhere(hough_slice > threshold_hough)
        for y, x in peaks:
            detected_circles.append((x, y, r))

    output_image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    for circle in detected_circles:
        cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

    cv2.imwrite("detected_circles_with_orientation.png", output_image)

    hough_space_2d = np.sum(hough_space, axis=2)

    hough_space_2d_norm = cv2.normalize(hough_space_2d, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("hough_space_2d.png", hough_space_2d_norm.astype(np.uint8))

    bounding_boxes = []
    for circle in detected_circles:
        x_center, y_center, radius = circle
        x1 = int(x_center - radius)
        y1 = int(y_center - radius)
        w = int(2 * radius)  # Width
        h = int(2 * radius)  # Height
        bounding_boxes.append(((x1, y1), (x1 + w, y1 + h), w, h))
    return bounding_boxes
