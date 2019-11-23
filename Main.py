import math
from typing import List, Tuple

import cv2
import numpy as np

RED = (0,0,255)
GREEN = (0,255,0)
PINK = (255,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
BLACK = (0,0,0)


def main():
    originalImage = cv2.imread('medium.jpg')
    grayImage = getGrayImage(originalImage)
    # imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    # imageShowWithWait("edgeImage", edgeImage)

    initial, finish = getInitialAndFinishArea(edgeImage)
    drawCircle(originalImage, initial, RED)
    drawCircle(originalImage, finish, GREEN)
    edgeImage = getEdgeImageWithoutCircles(edgeImage, initial, finish)
    # imageShowWithWait("edgeImage", edgeImage)

    lines = getMazeWalls(edgeImage)
    drawLinesOnImage(originalImage, lines, PINK)
    imageShowWithWait("lineImage", originalImage)


def getGrayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)


def getEdgeImage(image):
    return cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]


def getInitialAndFinishArea(image):
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
    # (x, y, r) = finishArea
    # drawCircle(originalImage, r, x, y)
    # (x, y, r) = initialArea
    # cv2.circle(originalImage, (x, y), r, (0, 0, 255), 4)
    return initialArea, finishArea


def drawCircle(image, circle, color):
    (x, y, r) = circle
    cv2.circle(image, (x, y), r, color, 4)


def getEdgeImageWithoutCircles(edgeImage, circle1, circle2):
    (x, y, r) = circle1
    cv2.circle(edgeImage, (x, y), int(r), (0, 0, 0), -4)
    (x, y, r) = circle2
    cv2.circle(edgeImage, (x, y), int(r), (0, 0, 0), -4)
    return edgeImage


def getMazeWalls(edgeImage):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 4  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    linesX, linesY = clusterWallsInOrientation(lines)
    lines = linesX
    lines.extend(linesY)
    return lines


def clusterWallsInOrientation(lines):
    horizontalWalls = []
    verticalWalls = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x1 - x2) ** 2 > (y1 - y2) ** 2):
                horizontalWalls.append(line)
            else:
                verticalWalls.append(line)
    return clusterHorizontalWalls(horizontalWalls), clusterVerticalWalls(verticalWalls)


def findAllCloseHorizontalWalls(baseWall, lines):
    x1Base, y1Base, x2Base, y2Base = baseWall[0]
    x1BaseAux = x1Base
    x2BaseAux = x2Base
    x1Base = min(x1BaseAux, x2BaseAux)
    x2Base = max(x1BaseAux, x2BaseAux)
    closeLines = [baseWall]
    maybeLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if math.fabs(y1Base - y1) < 20 and math.fabs(y2Base - y2) < 20:
            smallX = min(x1, x2)
            bigX = max(x1, x2)
            maybeLines.append(line)
            if x1Base <= smallX <= x2Base or x1Base <= bigX <= x2Base:
                closeLines.append(line)
                if x2Base < bigX:
                    x2Base = bigX
                    lines.extend(maybeLines)
                    maybeLines = []
                if x1Base > smallX:
                    x1Base = smallX
                    lines.extend(maybeLines)
                    maybeLines = []
    return closeLines


def findAllCloseVerticalWalls(baseWall, lines):
    x1Base, y1Base, x2Base, y2Base = baseWall[0]
    originalImage = cv2.imread('medium.jpg')
    # cv2.line(originalImage, (x1Base, y1Base), (x2Base, y2Base), PINK, 8)
    # delay = 10
    y1BaseAux = y1Base
    y2BaseAux = y2Base
    y1Base = min(y1BaseAux, y2BaseAux)
    y2Base = max(y1BaseAux, y2BaseAux)
    closeLines = [baseWall]
    maybeLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xMedium = (x1Base + x2Base) / 2
        rangeOfDistance = 20
        if xMedium - rangeOfDistance <= x1 <= xMedium + rangeOfDistance or xMedium - rangeOfDistance <= x2 <= xMedium + rangeOfDistance:
            smallY = min(y1, y2)
            bigY = max(y1, y2)
            maybeLines.append(line)
            if y1Base <= smallY <= y2Base or y1Base <= bigY <= y2Base:
                closeLines.append(line)
                if y2Base < bigY:
                    y2Base = bigY
                    lines.extend(maybeLines)
                    maybeLines = []
                if y1Base > smallY:
                    y1Base = smallY
                    lines.extend(maybeLines)
                    maybeLines = []

    #             cv2.line(originalImage, (x1, y1), (x2, y2), (0, 0, 255), 4)
    #             cv2.imshow("saidjd", originalImage)
    #             cv2.waitKey(delay)
    #         # a = 1 + 1
    #         else:
    #             cv2.line(originalImage, (x1, y1), (x2, y2), (0, 255, 0), 4)
    #             cv2.imshow("saidjd", originalImage)
    #             cv2.waitKey(delay)
    #
    #     else:
    #         cv2.line(originalImage, (x1, y1), (x2, y2), (255, 255, 255), 4)
    #         cv2.imshow("saidjd", originalImage)
    #         cv2.waitKey(delay)
    # cv2.waitKey(2000)

    return closeLines


def clusterHorizontalWalls(lines):
    clusteredLines = []
    while len(lines) != 0:
        line = lines[0]
        closeLines = findAllCloseHorizontalWalls(line, lines)
        lines = removeAll(lines, closeLines)
        clusteredLines.append(unifyCloseHorizontalLines(closeLines))
    return clusteredLines


def clusterVerticalWalls(lines):
    clusteredLines = []
    while len(lines) != 0:
        line = lines[0]
        closeLines = findAllCloseVerticalWalls(line, lines)
        lines = removeAll(lines, closeLines)
        clusteredLines.append(unifyCloseVerticalLines(closeLines))
    return clusteredLines


def removeAll(array, objectsThatWillBeDeleted):
    for object in objectsThatWillBeDeleted:
        i = 0
        while i < len(array):
            x1a, y1a, x2a, y2a = array[i][0]
            x1b, y1b, x2b, y2b = object[0]
            if x1a == x1b and x2a == x2b and y1a == y1b and y2a == y2b:
                del array[i]
            else:
                i += 1
    return array


def imageShowWithWait(windowName, image):
    cv2.imshow(windowName, image)
    cv2.waitKey(60000)


def unifyCloseHorizontalLines(lines):
    maxX, maxY, minX, minY = getExtremesOfLines(lines)
    return [[maxX, int((minY + maxY) / 2), minX, int((minY + maxY) / 2)]]


def unifyCloseVerticalLines(lines):
    maxX, maxY, minX, minY = getExtremesOfLines(lines)
    return [[int((minX + maxX) / 2), minY, int((minX + maxX) / 2), maxY]]


def getExtremesOfLines(lines):
    maxX = 0
    minX = float('inf')
    maxY = 0
    minY = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 > maxX:
            maxX = x1
        if x1 < minX:
            minX = x1
        if x2 > maxX:
            maxX = x2
        if x2 < minX:
            minX = x2

        if y1 > maxY:
            maxY = y1
        if y1 < minY:
            minY = y1
        if y2 > maxY:
            maxY = y2
        if y2 < minY:
            minY = y2
    return maxX, maxY, minX, minY


def drawLinesOnImage(image, lines, color):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, 5)


if __name__ == '__main__':
    main()
