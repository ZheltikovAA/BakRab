import pyglet
import numba
import numpy as np


class Human:
    def __init__(self, x, y, size, wall):
        vertices = [
            x, y,
            x, y + size,
            x + size, y + size,
            x + size, y,
        ]

        self.vertex_list = pyglet.graphics.vertex_list(4, ('v2i', vertices))

        self.x = self.start_pos_x = x
        self.y = self.start_pos_y = y
        self.size = size
        self.vertices = vertices
        self.new_x = 0
        self.new_y = 0
        self.num_rays = 4
        self.angle_between_rays = 360.0 / self.num_rays
        self.max_ray_length = 50.0

        self.score = 0
        self.keys = {
            pyglet.window.key.LEFT: False,
            pyglet.window.key.RIGHT: False,
            pyglet.window.key.UP: False,
            pyglet.window.key.DOWN: False,
        }
        self.wall = wall
        self.color = (255, 0, 0)

        self.ray_color = (255, 123, 10)

    def move_left(self):
        self.new_x -= 1

    def move_right(self):
        self.new_x += 1

    def move_up(self):
        self.new_y += 1

    def move_down(self):
        self.new_y -= 1

    def draw(self):
        pyglet.gl.glColor3f(*self.color)
        self.vertex_list.draw(pyglet.gl.GL_QUADS)
        # self.draw_rays()

    def restart(self):
        self.x = self.start_pos_x
        self.y = self.start_pos_y
        self.score += 1

    def check_collision(self, x, y):
        # Проверяем столкновение с каждой стеной
        for i, row in enumerate(self.wall.matrix):
            for j, value in enumerate(row):
                if value == 1 or value == 2:
                    wall_x = j * self.wall.cell_width
                    wall_y = (self.wall.matrix_height - i - 1) * self.wall.cell_height
                    if x + self.size > wall_x and x < wall_x + self.wall.cell_width and \
                            y + self.size > wall_y and y < wall_y + self.wall.cell_height:
                        return True
        return False

    def update(self, dt):
        # изменяем координаты квадрата на заданный вектор
        dx = (self.keys[pyglet.window.key.RIGHT] - self.keys[pyglet.window.key.LEFT]) * 10
        dy = (self.keys[pyglet.window.key.UP] - self.keys[pyglet.window.key.DOWN]) * 10

        x = self.x + dx
        y = self.y + dy
        # # Параметры для модели
        # x = self.new_x
        # y = self.new_y

        if (10 >= y >= 0) and (1255 <= x <= 1260):
            self.restart()
            print(self.score)
            return

        if 0 <= x <= self.wall.width - self.size and 0 <= y <= self.wall.height - self.size \
                and self.check_collision(x, y):
            self.x = x
            self.y = y

        # обновляем список вершин квадрата с новыми координатами
        vertices = [
            self.x, self.y,
            self.x, self.y + self.size,
            self.x + self.size, self.y + self.size,
            self.x + self.size, self.y,
        ]
        self.vertex_list.vertices = vertices

    def draw_rays(self):
        # Позиция человека
        global wall_x, wall_y
        x = self.x + self.size / 2
        y = self.y + self.size / 2

        # Отрисовка лучей
        for i in numba.prange(self.num_rays):
            # Угол текущего луча
            angle = np.radians(i * self.angle_between_rays)

            # Координаты направления луча
            ray_dir_x = np.cos(angle)
            ray_dir_y = np.sin(angle)

            # Начальные координаты луча
            ray_start_x = x
            ray_start_y = y
            distance = [0] * 4
            # Итерация по лучу для нахождения стены
            for ray_length in numba.prange(int(self.max_ray_length)):
                # Координаты текущей точки на луче
                ray_x = ray_start_x + ray_dir_x * ray_length
                ray_y = ray_start_y + ray_dir_y * ray_length
                if ray_x < 0 or ray_x > self.wall.width or ray_y < 0 or ray_y > self.wall.height:
                    # Отрисовка луча
                    pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', (x, y, ray_x, ray_y)),
                                         ('c3B', self.ray_color * 2))
                    break
                # Проверка столкновения с каждой стеной
                for i, row in enumerate(self.wall.matrix):
                    for j, value in enumerate(row):
                        if value == 0:
                            wall_x = j * self.wall.cell_width
                            wall_y = (self.wall.matrix_height - i - 1) * self.wall.cell_height
                            if wall_x <= ray_x <= wall_x + self.wall.cell_width and \
                                    wall_y <= ray_y <= wall_y + self.wall.cell_height:
                                # Отрисовка луча
                                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', (x, y, ray_x, ray_y)),
                                                     ('c3B', self.ray_color * 2))

                                # Отрисовка точки пересечения
                                pyglet.graphics.draw(1, pyglet.gl.GL_POINTS, ('v2f', (ray_x, ray_y)),
                                                     ('c3B', self.ray_color))
                                break

                    # Прерываем внутренний цикл, если найдена стена
                    if wall_x <= ray_x <= wall_x + self.wall.cell_width and \
                            wall_y <= ray_y <= wall_y + self.wall.cell_height:
                        break

                # Прерываем внешний цикл, если найдена стена
                if wall_x <= ray_x <= wall_x + self.wall.cell_width and \
                        wall_y <= ray_y <= wall_y + self.wall.cell_height:
                    break
