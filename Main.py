import heapq
import math
from time import sleep
import cv2
import numpy as np
import os

RED = (0, 0, 255)
GREEN = (0, 255, 0)
PINK = (255, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)
# IMAGE_NAME = 'lab1.jpg'
IMAGE_NAME = 'medium.jpg'
# IMAGE_NAME = 'easy.jpg'
# IMAGE_NAME = 'hard.png'
IMAGE_NAME = 'lab3.png'
SHOW_LINES_AS_GROUPING = False
roundNumber = 1
roundNumber2 = 20
minDif = 1000000

globalMin = 0
globalMinY = 0


def main():
    global roundNumber2
    originalImage = cv2.imread(IMAGE_NAME)
    grayImage = getGrayImage(originalImage)
    imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    imageShowWithWait("edgeImage", edgeImage)
    cv2.imwrite("edgeImage" + IMAGE_NAME, edgeImage)
    initial, finish = getInitialAndFinishArea(edgeImage)
    drawCircle(originalImage, initial, RED)
    drawCircle(originalImage, finish, GREEN)
    edgeImage = getEdgeImageWithoutCircles(edgeImage, initial, finish)
    # imageShowWithWait("edgeImageWithoutCircles", edgeImage)
    cv2.imwrite("edgeImageWithoutCircles" + IMAGE_NAME, edgeImage)

    linesX, linesY = getMazeWalls(edgeImage)
    drawLinesOnImage(originalImage, linesX, CYAN)
    drawLinesOnImage(originalImage, linesY, CYAN)
    imageShowWithWait("lineImage", originalImage, 10)
    cv2.imwrite("lineImage" + IMAGE_NAME, originalImage)

    mazeMatrix, start, end, blockSize, minXY, minYX = getMazeMatrix(linesX, linesY, initial, finish, originalImage)
    print(mazeMatrix)
    print(len(mazeMatrix), len(mazeMatrix[0]), roundNumber2)
    solutionMatrix = solveMaze(mazeMatrix, start, end)
    originalImage = cv2.imread(IMAGE_NAME)
    drawSolution(originalImage, solutionMatrix, blockSize, minXY, minYX, PINK)
    cv2.imwrite("solution" + IMAGE_NAME, originalImage)

class Cell(object):
    def __init__(self, x, y, reachable):
        """
        Initialize new cell

        @param x cell x coordinate
        @param y cell y coordinate
        @param reachable is cell reachable? not a wall?
        """
        self.reachable = reachable
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f


class AStar(object):
    def __init__(self):
        self.opened = []
        heapq.heapify(self.opened)
        self.closed = set()
        self.cells = []
        self.grid_height = 6
        self.grid_width = 6
        self.start = None
        self.end = None

    def init_grid(self, grid):

        self.grid_height = len(grid)
        self.grid_width = len(grid[0])
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if grid[y][x] == 1:
                    reachable = False
                else:
                    reachable = True
                self.cells.append(Cell(x, y, reachable))
                if grid[y][x] == 2:
                    if self.start == None:
                        self.start = self.get_cell(x, y)
                    else:
                        self.end = self.get_cell(x, y)

    def get_heuristic(self, cell):
        """
        Compute the heuristic value H for a cell: distance between
        this cell and the ending cell multiply by 10.

        @param cell
        @returns heuristic value H
        """
        return 10 * (abs(cell.x - self.end.x) + abs(cell.y - self.end.y))

    def get_cell(self, x, y):
        """
        Returns a cell from the cells list

        @param x cell x coordinate
        @param y cell y coordinate
        @returns cell
        """
        return self.cells[x * self.grid_height + y]

    def get_adjacent_cells(self, cell):
        """
        Returns adjacent cells to a cell. Clockwise starting
        from the one on the right.

        @param cell get adjacent cells for this cell
        @returns adjacent cells list
        """
        cells = []
        if cell.x < self.grid_width - 1:
            cells.append(self.get_cell(cell.x + 1, cell.y))
        if cell.y > 0:
            cells.append(self.get_cell(cell.x, cell.y - 1))
        if cell.x > 0:
            cells.append(self.get_cell(cell.x - 1, cell.y))
        if cell.y < self.grid_height - 1:
            cells.append(self.get_cell(cell.x, cell.y + 1))
        return cells

    def display_path(self):
        cell = self.end
        solution = []
        while cell.parent is not self.start:
            cell = cell.parent
            solution.append((cell.x, cell.y))
            print( 'path: cell: %d,%d' % (cell.x, cell.y))
        return solution

    def update_cell(self, adj, cell):
        """
        Update adjacent cell

        @param adj adjacent cell to current cell
        @param cell current cell being processed
        """
        adj.g = cell.g + 10
        adj.h = self.get_heuristic(adj)
        adj.parent = cell
        adj.f = adj.h + adj.g

    def process(self):
        # add starting cell to open heap queue
        heapq.heappush(self.opened, (self.start.f, self.start))
        while len(self.opened):
            # pop cell from heap queue
            f, cell = heapq.heappop(self.opened)
            # add cell to closed list so we don't process it twice
            self.closed.add(cell)
            # if ending cell, display found path
            if cell is self.end:
                self.display_path()
                break
            # get adjacent cells for cell
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.reachable and adj_cell not in self.closed:
                    if (adj_cell.f, adj_cell) in self.opened:
                        # if adj cell in open list, check if current path is
                        # better than the one previously found for this adj
                        # cell.
                        if adj_cell.g > cell.g + 10:
                            self.update_cell(adj_cell, cell)
                    else:
                        self.update_cell(adj_cell, cell)
                        # add adj cell to open list
                        heapq.heappush(self.opened, (adj_cell.f, adj_cell))

def getGrayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)


def getEdgeImage(image):
    return cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]


def getInitialAndFinishArea(image):
    minDimension = min(len(image), len(image[0]))
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 200, None, 30, 15, int(minDimension * 0.0015), int(minDimension * 0.1))
    # circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 200, None, 30, 15, 1, 14)
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

    cv2.circle(edgeImage, (x, y), int(r * 1.05), (0, 0, 0), -1)
    (x, y, r) = circle2
    cv2.circle(edgeImage, (x, y), int(r * 1.05), (0, 0, 0), -1)
    return edgeImage


def getMazeWalls(edgeImage):
    imageWidth = len(edgeImage[0])
    imageHeight = len(edgeImage)
    minDimension = min(imageHeight, imageWidth)
    global roundNumber2
    roundNumber2 = minDimension * 0.03
    rho = roundNumber2/6  # distance resolution in pixels of the Hough grid
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
        if yMedium - rangeOfDistance <= y1 <= yMedium + rangeOfDistance and yMedium - rangeOfDistance <= y2 <= yMedium + rangeOfDistance:
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


def imageShowWithWait(windowName, image, time=1000):
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


def getMazeMatrix(linesX, linesY, initial, finish, image):

    linesX.sort(key=lambda x: x[0][1], reverse=False)
    linesY.sort(key=lambda x: x[0][0], reverse=False)

    # # maxX, maxYX, minX, minYX = getExtremesOfLines(linesX)
    # # maxXY, maxY, minXY, minY = getExtremesOfLines(linesY)
    # maxX = max(maxX, maxXY)
    # maxY = max(maxY, maxYX)
    blockSize = int(minDif * 1)
    tamX = math.ceil(len(image[0]) / blockSize) + 1
    tamY = math.ceil(len(image) / blockSize) + 1
    maze = np.zeros((tamY, tamX))

    for line in linesX:
        x1, y, x2, _ = line[0]
        i = int(round((y) / blockSize))
        jMin = int(round((x2) / blockSize))
        jMax = int(round(x1/blockSize))
        for j in range(jMin, jMax):
            maze[i][j] = 1

    for line in linesY:
        x, y1, _, y2 = line[0]
        j = int(round((x) / blockSize))
        iMin = int(round((y1) / blockSize))
        iMax = int(round((y2) / blockSize))
        for i in range(iMin, iMax):
            maze[i][j] = 1
    maze[int(round((initial[1]) / blockSize))][int(round((initial[0]) / blockSize))] = 2
    maze[int(round((finish[1]) / blockSize))][int(round((finish[0]) / blockSize))] = 2

    return \
        maze, \
        [int(round((initial[1] ) / blockSize)), int(round((initial[0] ) / blockSize))], \
        [int(round((finish[1] ) / blockSize)), int(round((finish[0] ) / blockSize))], \
        blockSize, \
        0, \
        1000


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
        drawLine(image, line, color, 4)


def drawLine(image, line, color, lineWidth=5):
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), color, lineWidth)


def solveMaze(maze, start, end):
    # sol = np.zeros((len(maze), len(maze[0])))
    # sol[start[1]][start[0]] = 2
    # hasSolution = solveMazeUtil(maze, start[1], start[0], sol, end[1], end[0])
    # if hasSolution == False:
    #     print("Solution doesn't exist")
    #     quit(-1)
    # return sol
    aStar = AStar()
    aStar.init_grid(maze)
    aStar.process()
    return aStar.display_path()


def solveMazeUtil(maze, x, y, sol, finalX, finalY):
    if x >= len(maze[0]) or y >= len(maze):
        return False
    if x == finalX and y == finalY:
        sol[y][x] = 2
        return True
    if maze[y][x] == 1:
        sol[y][x] = 1
        return False
    if sol[y][x] == 3:
        return False
    if sol[y][x] == 0:
        sol[y][x] = 3

    hasSolution1 = solveMazeUtil(maze, x, y - 1, sol, finalX, finalY)
    if hasSolution1:
        return True
    hasSolution2 = solveMazeUtil(maze, x, y + 1, sol, finalX, finalY)
    if hasSolution2:
        return True

    hasSolution3 = solveMazeUtil(maze, x + 1, y, sol, finalX, finalY)
    if hasSolution3:
        return True
    hasSolution4 = solveMazeUtil(maze, x - 1, y, sol, finalX, finalY)
    if hasSolution4:
        return True
    return False


def drawSolution(originalImage, solution, blockSize, minXYa, minYXa, color):
    # simplifySolution(solutionMatrix)
    # print(solutionMatrix)
    # print(simplifySolution(solutionMatrix))
    maxX = len(originalImage[0]) - 1
    maxY = len(originalImage) - 1
    halfBlockSize = int(blockSize / 2) * 0
    for i in range(1, len(solution)):
        before = solution[i - 1]
        current = solution[i]
        x = current[0]
        y = current[1]
        xBefore = before[0]
        yBefore = before[1]
        # if not before[0] == current[0]:
        drawLine(originalImage, [
                                [x * blockSize + halfBlockSize, y * blockSize + halfBlockSize ,
                                 xBefore * blockSize + halfBlockSize , yBefore * blockSize + halfBlockSize ]], color)
        # else:
        #     drawLine(originalImage, [
        #         [x * blockSize + halfBlockSize, y * blockSize + halfBlockSize,
        #          xBefore * blockSize + halfBlockSize, yBefore * blockSize + halfBlockSize]], color)

    # for i in range(0, maxX):
    #     for j in range(0, maxY):
    #         if solutionMatrix[j][i] >= 2:
    #             if solutionMatrix[j + 1][i] >= 2:
    #                 drawLine(originalImage, [
    #                     [i * blockSize + minXY + halfBlockSize, j * blockSize + halfBlockSize + minYX,
    #                      i * blockSize + halfBlockSize + minXY, (j + 1) * blockSize + halfBlockSize + minYX]], color)
    #             if solutionMatrix[j][i + 1] >= 2:
    #                 drawLine(originalImage, [
    #                     [i * blockSize + minXY + halfBlockSize, j * blockSize + halfBlockSize + minYX,
    #                      (i + 1) * blockSize + halfBlockSize + minXY, j * blockSize + halfBlockSize + minYX]], color)
    imageShowWithWait("solution", originalImage, 100)
    # sleep(100)


def simplifySolution(solutionMatrix):
    for i in range(1, len(solutionMatrix[0]) - 1):
        for j in range(1, len(solutionMatrix) - 1):
            if solutionMatrix[j + 1][i] == 3 and solutionMatrix[j][i + 1] == 3 and solutionMatrix[j + 1][i + 1] == 3 and \
                    solutionMatrix[j][i] == 3:
                iAux = i + 1
                jAux = j + 1
                stillSquare = True
                while (stillSquare):
                    iAux += 1
                    for a in range(j, jAux + 1):
                        if stillSquare:
                            stillSquare = solutionMatrix[a][iAux] == 3

                stillSquare = True
                while (stillSquare):
                    jAux += 1
                    for a in range(i, iAux):
                        if stillSquare:
                            stillSquare = solutionMatrix[jAux][a] == 3
                # for y in range(j, jAux ):
                #     for x in range(i, iAux ):
                #         solutionMatrix[y][x] = 4
                reduceRedudantPart(i, iAux, j, jAux, solutionMatrix)
    # for i in range(1, len(solutionMatrix[0]) - 1):
    #     for j in range(1, len(solutionMatrix) - 1):
    #         if solutionMatrix[j][i] == 3:
                # iAux = i + 1
                # while(solutionMatrix [j][iAux] == 3):
                #     iAux += 1
                # # reduceRedudantPart(i, iAux, j, j +1, solutionMatrix)
                #
                # jAux = j + 1
                # while(solutionMatrix [jAux][i] == 3):
                #     jAux += 1
                # reduceRedudantPart(i, i+1, j, jAux, solutionMatrix)

    return solutionMatrix


def reduceRedudantPart(i, iAux, j, jAux, solutionMatrix):
    HAS_SOLUTION_TOP_LEFT = solutionMatrix[j - 1][i] >= 2 or solutionMatrix[j][i - 1] >= 2
    HAS_SOLUTION_TOP_RIGHT = solutionMatrix[j - 1][iAux - 1] >= 2 or solutionMatrix[j][iAux] >= 2
    HAS_SOLUTION_BOTTOM_LEFT = solutionMatrix[jAux][i] >= 2 or solutionMatrix[jAux - 1][i - 1] >= 2
    HAS_SOLUTION_BOTTOM_RIGHT = solutionMatrix[jAux][iAux - 1] >= 2 or solutionMatrix[jAux - 1][iAux] >= 2
    totalTrue = 0
    if HAS_SOLUTION_TOP_LEFT:
        totalTrue += 1
    if HAS_SOLUTION_TOP_RIGHT:
        totalTrue += 1
    if HAS_SOLUTION_BOTTOM_LEFT:
        totalTrue += 1
    if HAS_SOLUTION_BOTTOM_RIGHT:
        totalTrue += 1
    if totalTrue == 2:
        if HAS_SOLUTION_BOTTOM_LEFT and HAS_SOLUTION_BOTTOM_RIGHT:
            for y in range(j, jAux - 1):
                for x in range(i, iAux):
                    solutionMatrix[y][x] = 0
        elif HAS_SOLUTION_TOP_LEFT and HAS_SOLUTION_TOP_RIGHT:
            for y in range(j + 1, jAux):
                for x in range(i, iAux):
                    solutionMatrix[y][x] = 0
        elif HAS_SOLUTION_TOP_RIGHT and HAS_SOLUTION_BOTTOM_RIGHT:
            for y in range(j, jAux):
                for x in range(i, iAux - 1):
                    solutionMatrix[y][x] = 0
        elif HAS_SOLUTION_TOP_LEFT and HAS_SOLUTION_BOTTOM_LEFT:
            for y in range(j, jAux):
                for x in range(i + 1, iAux):
                    solutionMatrix[y][x] = 0
        elif HAS_SOLUTION_BOTTOM_LEFT and HAS_SOLUTION_TOP_RIGHT:
            for y in range(j, jAux - 1):
                for x in range(i, iAux - 1):
                    solutionMatrix[y][x] = 0
        elif HAS_SOLUTION_BOTTOM_RIGHT and HAS_SOLUTION_TOP_LEFT:
            for y in range(j + 1, jAux):
                for x in range(i, iAux - 1):
                    solutionMatrix[y][x] = 0
    if totalTrue < 2:
        for y in range(j, jAux):
            for x in range(i, iAux):
                solutionMatrix[y][x] = 0


if __name__ == '__main__':
    main()


