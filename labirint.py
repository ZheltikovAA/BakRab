import random
import time
import psutil as ps


# Начальные координаты
start_x = 0
start_y = 0
# Стек для отслеживания позиций
stack = [(start_x, start_y)]


# Генерация лабиринта

def labirint(maze, width_lab, height_lab):

    # loadavg = ps.cpu_percent(interval=None, percpu=True)
    # print(loadavg, "<-- In start time")
    # i = 0
    stack = [(0, 0)]
    while stack:
        x, y = stack[-1]
        maze[y][x] = 1
        neighbors = [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]
        random.shuffle(neighbors)
        found = False
        for nx, ny in neighbors:
            if 0 <= nx < width_lab and 0 <= ny < height_lab and maze[ny][nx] == 0:
                maze[(ny + y) // 2][(nx + x) // 2] = 1
                maze[ny][nx] = 1
                stack.append((nx, ny))
                found = True
                break
        if not found:
            stack.pop()
    maze[height_lab - 1][width_lab - 2] = 2
    # loadavg2 = ps.cpu_percent(interval=None, percpu=True)
    # print("All time spent : {} , loadavg {}".format(execution_time, loadavg2))
    # print(maze)
    return [maze]
