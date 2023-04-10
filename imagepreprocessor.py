import cv2
import numpy as np


def processedImage(frame):
    # transform image to HSV color space:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create color filter to filter out white(-ish) areas from the image:
    sensitivity = 35

    lower_white = np.array([0, 0, 255 - sensitivity], dtype=np.uint8)
    upper_white = np.array([255, sensitivity, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # the res frame is still has three dimensions: red, green and blue -> create a grayscale image:

    h, s, v = cv2.split(res)
    grayscale = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

    # create a threshold filter for the image:

    thresh, binary = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)

    # create perspective warp of the image:

    pts1 = np.float32([[280, 280], [345, 280], [150, 360], [440, 360]])
    pts2 = np.float32([[130, 100], [460, 100], [150, 360], [440, 360]])

    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_warped_image = cv2.warpPerspective(binary, transformation_matrix, (binary.shape[1], binary.shape[0]))

    return perspective_warped_image
