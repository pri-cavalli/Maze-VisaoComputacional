from typing import List, Tuple

import cv2
import numpy as np

def main():
    originalImage = cv2.imread('medium.jpg')

    grayImage = getGrayImage(originalImage)
    # imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    # imageShowWithWait("edgeImage", edgeImage)

    initial, finish = getInitialAndFinishArea(edgeImage, originalImage)

    edgeImage = getEdgeImageWithoutCircles(edgeImage, initial, finish)
    # imageShowWithWait("edgeImage", edgeImage)

    getMazeWalls(edgeImage, originalImage)
    imageShowWithWait("lineImage", originalImage)

def getGrayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)

def getEdgeImage(image):
    return cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

def getInitialAndFinishArea(image, originalImage):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 200, None, 30, 15, 5, 50)
    if len(circles) > 2:
        raise Exception("Achou mais de 2 cisúlos na imagem")
    print(circles)
    if circles[0][0][2] > circles[0][1][2]:
        finishArea = circles[0][0]
        initialArea = circles[0][1]
    else:
        finishArea = circles[0][1]
        initialArea = circles[0][0]
    (x, y, r) = finishArea
    cv2.circle(originalImage, (x, y), r, (0, 255, 0), 4)
    (x, y, r) = initialArea
    cv2.circle(originalImage, (x, y), r, (0, 0, 255), 4)
    return initialArea, finishArea

def getEdgeImageWithoutCircles(edgeImage, circle1, circle2):
    (x, y, r) = circle1
    cv2.circle(edgeImage, (x, y), int(r+5), (0, 0, 0), -4)
    (x, y, r) = circle2
    cv2.circle(edgeImage, (x, y), int(r+5), (0, 0, 0), -4)
    return edgeImage


def getMazeWalls(edgeImage, originalImage):
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 20  # angular resolution in radians of the Hough grid
    threshold = 20 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(originalImage, (x1, y1), (x2, y2), (200, 200, 0), 1)
    return lines

def imageShowWithWait(windowName, image):
    cv2.imshow(windowName, image)
    cv2.waitKey(6000)

if __name__ == '__main__':
    main()
