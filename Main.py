import math

import cv2
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 255, 0)
PINK = (255, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)
# IMAGE_NAME = 'medium.jpg'
IMAGE_NAME = 'medium.jpg'
SHOW_LINES_AS_GROUPING = False
# SHOW_LINES_AS_GROUPING = True
roundNumber = 1
roundNumber2 = 15
minDif = 1000000

def main():
    originalImage = cv2.imread(IMAGE_NAME)
    grayImage = getGrayImage(originalImage)
    #imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    #imageShowWithWait("edgeImage", edgeImage)

    initial, finish = getInitialAndFinishArea(edgeImage)
    drawCircle(originalImage, initial, RED)
    drawCircle(originalImage, finish, GREEN)
    edgeImage = getEdgeImageWithoutCircles(edgeImage, initial, finish)
    #imageShowWithWait("edgeImageWithoutCircles", edgeImage)

    linesX, linesY = getMazeWalls(edgeImage)
    drawLinesOnImage(originalImage, linesX, PINK)
    drawLinesOnImage(originalImage, linesY, PINK)
    mazeMatrix = getMazeMatrix(linesX, linesY, initial, finish);
    #imageShowWithWait("lineImage", originalImage, 1005555)



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

    cv2.circle(edgeImage, (x, y), int(r*1.05), (0, 0, 0), -1)
    (x, y, r) = circle2
    cv2.circle(edgeImage, (x, y), int(r*1.05), (0, 0, 0), -1)
    return edgeImage


def getMazeWalls(edgeImage):
    imageWidth = len(edgeImage[0])
    imageHeight = len(edgeImage)
    minDimension = min(imageHeight, imageWidth)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 4  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = int(minDimension * .08)  # minimum number of pixels making up a line
    max_line_gap = int(minDimension * .03)  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    return clusterWallsInOrientation(lines, minDimension)


def clusterWallsInOrientation(lines, minDimension):
    horizontalWalls = []
    verticalWalls = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x1 - x2) ** 2 > (y1 - y2) ** 2):
                horizontalWalls.append(line)
            else:
                verticalWalls.append(line)
    return clusterHorizontalWalls(horizontalWalls, minDimension), clusterVerticalWalls(verticalWalls, minDimension)


def findAllCloseHorizontalWalls(baseWall, lines, minDimension):
    x1Base, y1Base, x2Base, y2Base = baseWall[0]
    image = cv2.imread(IMAGE_NAME)
    drawLine(image, baseWall, PINK, 10)
    x1BaseAux = x1Base
    x2BaseAux = x2Base
    x1Base, x2Base = getMinAndMax(x1Base, x2Base)
    x2Base = max(x1BaseAux, x2BaseAux)
    closeLines = [baseWall]
    maybeLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        yMedium = (y1Base + y2Base) / 2
        rangeOfDistance = minDimension * 0.04
        if yMedium - rangeOfDistance <= y1 <= yMedium + rangeOfDistance and yMedium - rangeOfDistance <= y2 <= yMedium + rangeOfDistance :
            smallX, bigX = getMinAndMax(x1, x2)
            maybeLines.append(line)
            if x1Base - rangeOfDistance <= smallX <= x2Base + rangeOfDistance or x1Base - rangeOfDistance <= bigX <= x2Base + rangeOfDistance:
                closeLines.append(line)
                if x2Base < bigX:
                    x2Base = bigX
                    lines.extend(maybeLines)
                    maybeLines = []
                if x1Base > smallX:
                    x1Base = smallX
                    lines.extend(maybeLines)
                    maybeLines = []
                if SHOW_LINES_AS_GROUPING:
                    drawLineAndShow(image, line, CYAN)
            elif SHOW_LINES_AS_GROUPING:
                drawLineAndShow(image, line, GREEN)
        elif SHOW_LINES_AS_GROUPING:
            drawLineAndShow(image, line, WHITE)
    return closeLines


def getMinAndMax(x1, x2):
    smallX = min(x1, x2)
    bigX = max(x1, x2)
    return smallX, bigX


def findAllCloseVerticalWalls(baseWall, lines, minDimension):
    x1Base, y1Base, x2Base, y2Base = baseWall[0]
    image = cv2.imread(IMAGE_NAME)
    drawLine(image, baseWall, PINK, 10)
    y1BaseAux = y1Base
    y2BaseAux = y2Base
    y1Base = min(y1BaseAux, y2BaseAux)
    y2Base = max(y1BaseAux, y2BaseAux)
    closeLines = [baseWall]
    maybeLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xMedium = (x1Base + x2Base) / 2
        rangeOfDistance = minDimension * 0.04
        if xMedium - rangeOfDistance <= x1 <= xMedium + rangeOfDistance and xMedium - rangeOfDistance <= x2 <= xMedium + rangeOfDistance:
            smallY = min(y1, y2)
            bigY = max(y1, y2)
            maybeLines.append(line)
            if y1Base - rangeOfDistance <= smallY <= y2Base + rangeOfDistance or y1Base - rangeOfDistance <= bigY <= y2Base + rangeOfDistance:
                closeLines.append(line)
                if y2Base < bigY:
                    y2Base = bigY
                    lines.extend(maybeLines)
                    maybeLines = []
                if y1Base > smallY:
                    y1Base = smallY
                    lines.extend(maybeLines)
                    maybeLines = []
                if SHOW_LINES_AS_GROUPING:
                    drawLineAndShow(image, line, CYAN)
            elif SHOW_LINES_AS_GROUPING:
                drawLineAndShow(image, line, GREEN)
        elif SHOW_LINES_AS_GROUPING:
            drawLineAndShow(image, line, WHITE)
    return closeLines


def drawLineAndShow(image, line, color):
    delay = 5
    drawLine(image, line, color)
    cv2.imshow("lines", image)


def clusterHorizontalWalls(lines, minDimension):
    clusteredLines = []
    while len(lines) != 0:
        line = lines[0]
        closeLines = findAllCloseHorizontalWalls(line, lines, minDimension)
        lines = removeAll(lines, closeLines)
        clusteredLines.append(unifyCloseHorizontalLines(closeLines))
    return padronizeHeightOfHorizontalLines(clusteredLines)


def clusterVerticalWalls(lines, minDimension):
    clusteredLines = []
    while len(lines) != 0:
        line = lines[0]
        closeLines = findAllCloseVerticalWalls(line, lines, minDimension)
        lines = removeAll(lines, closeLines)
        clusteredLines.append(unifyCloseVerticalLines(closeLines))
    return padronizeXverticalLines(clusteredLines)


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


def imageShowWithWait(windowName, image, time = 1000):
    cv2.imshow(windowName, image)
    cv2.waitKey(time)


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

def getMazeMatrix(linesX, linesY, initial, finish):
    linesX.sort(key=lambda x: x[0][1], reverse=False)
    linesY.sort(key=lambda x: x[0][0], reverse=False)

    maxX, _, minX, _ = getExtremesOfLines(linesX)
    _, maxY, _, minY = getExtremesOfLines(linesY)

    blockSize = int(minDif * 1)
    tamX = math.ceil((maxX - minX) / blockSize)
    tamY = math.ceil((maxY - minY) / blockSize)
    maze = np.zeros((tamY, tamX))


    for line in linesX:
        x1, y, x2, _ = line[0]
        i = int((y - minY) / blockSize)
        j = 0
        while(j < tamX):
            if (x2 <= minX + j * blockSize <= x1) or j == 0 or j == tamY-1:
                maze[i][j] = 1
            elif maze[i][j] != 1:
                maze[i][j] = 0
            j += 1


    for line in linesY:
        x, y1, _, y2 = line[0]
        j = int((x - minX) / blockSize)
        i = 0
        while(i < tamY):
            if (y1 <= minY + i * blockSize <= y2) or i == 0 or i == tamX-1:
                maze[i][j] = 1
            elif maze[i][j] != 1:
                maze[i][j] = 0
            i += 1
    maze[int(initial[1]/blockSize)][int(initial[0]/blockSize)] = 2
    maze[int(finish[1]/blockSize)][int(finish[0]/blockSize)] = 2
    # print(maze)
    x = 0

    maxY = len(maze[0]) - 1
    maxX = len(maze) - 1
    for i in range(0, maxX):
        for j in range(0, maxY):
            if maze[i][j] == 0:
                print("  ", end='')
            elif maze[i][j] == 1:
                print("++", end='')
            elif maze[i][j] == 2:
                print("()", end='')


        print(" ", x)
        x +=1
    print(linesY)
    print(linesX)


def padronizeXverticalLines(linesY):
    linesY.sort(key=lambda x: x[0][0], reverse=False)

    lastX = 0
    global minDif
    for line in linesY:
        x, y1, _, y2 = line[0]
        if x - lastX < roundNumber2:
            line[0] = (lastX, y1, lastX, y2)
        elif x - lastX < minDif:
            minDif = x - lastX

        lastX = x
    return linesY


def padronizeHeightOfHorizontalLines(linesX):
    global minDif
    linesX.sort(key=lambda x: x[0][1], reverse=False)
    lastY = 0

    for line in linesX:
        x1, y, x2, _ = line[0]
        if y - lastY < roundNumber2:
            line[0] = (x1, lastY, x2, lastY)
        elif y - lastY < minDif:
            minDif = y - lastY
        lastY = y
    return linesX


def drawLinesOnImage(image, lines, color):
    for line in lines:
        drawLine(image, line, color, 2)


def drawLine(image, line, color, lineWidth = 5):
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), color, lineWidth)


if __name__ == '__main__':
    main()
