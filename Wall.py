import pyglet
import random
import NNConstructMaze
import labirint
import numba


class Wall:
    def __init__(self, width_lab, height_lab, width, height):
        matrix = [[0] * width_lab for _ in range(height_lab)]
        self.matrix = labirint.labirint(matrix, width_lab, height_lab)
        # self.matrix = NNConstructMaze.get_maze_nn()
        print(len(self.matrix), " ", len(self.matrix[0]))
        self.matrix_height = height_lab
        self.matrix_width = width_lab
        # Размеры ячейки
        self.cell_width = width // len(self.matrix[0])
        self.cell_height = height // len(self.matrix)
        self.width = width
        self.height = height

    def draw(self):
        batch = pyglet.graphics.Batch()

        for i, row in enumerate(self.matrix):
            for j, value in enumerate(row):
                if value == 1:
                    # Отрисовка заполненной ячейки
                    x = j * self.cell_width
                    y = (
                                    len(self.matrix) - i - 1) * self.cell_height  # Инвертируем направление отрисовки по y для правильного расположения
                    vertex_list = batch.add(4, pyglet.gl.GL_QUADS, None,
                                            ('v2i',
                                             [x, y, x, y + self.cell_height, x + self.cell_width, y + self.cell_height,
                                              x + self.cell_width,
                                              y]),
                                            ('c3B', [255, 255, 255] * 4))  # Цвет заполненной ячейки (белый)
                if value == 2:
                    # Отрисовка заполненной ячейки
                    x = j * self.cell_width
                    y = (
                                len(self.matrix) - i - 1) * self.cell_height  # Инвертируем направление отрисовки по y для правильного расположения
                    vertex_list = batch.add(4, pyglet.gl.GL_QUADS, None,
                                            ('v2i',
                                             [x, y, x, y + self.cell_height, x + self.cell_width, y + self.cell_height,
                                              x + self.cell_width,
                                              y]),
                                            ('c3B', [0, 255, 0] * 4))  # Цвет заполненной ячейки (белый)
        batch.draw()

    def get_wall_data(self):
        wall_data = []
        for i in range(self.matrix_height):
            for j in range(self.matrix_width):
                if self.matrix[i][j] == 1:
                    wall_data.append(j)
                    wall_data.append(i)

        return wall_data
# создаем нужные рамки для окна
