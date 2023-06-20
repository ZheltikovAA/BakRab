import pyglet
import numba
import numpy as np
import Wall

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
        self.step_x = self.x
        self.step_y = self.y
        self.score = 0
        self.keys = {
            pyglet.window.key.LEFT: False,
            pyglet.window.key.RIGHT: False,
            pyglet.window.key.UP: False,
            pyglet.window.key.DOWN: False,
        }
        self.keys_AI = [0, 0, 0, 0]
        self.wall = wall
        self.color = (255, 0, 0)

        self.ray_color = (255, 123, 10)

    # Направление будем задавать массивом [up, down, right, left]

    def draw(self):
        pyglet.gl.glColor3f(*self.color)
        self.vertex_list.draw(pyglet.gl.GL_QUADS)

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

    def player_ai_step(self, action):
        self.move_ai(action)

        reward = -0.01
    #reward
        if self.step_x == self.x and self.step_y == self.y:
            reward = -0.05
            return reward, self.score

        if self.finish():
            reward = 10
            return reward, self.score

        return reward, self.score

    def move_ai(self, action):
        self.keys_AI = action

        dx = (self.keys_AI[2] - self.keys_AI[3]) * 10
        dy = (self.keys_AI[0] - self.keys_AI[1]) * 10

        self.keys_AI = [0, 0, 0, 0]

        self.step_x = self.x + dx
        self.step_y = self.y + dy


    def finish(self):
        if (10 >= self.step_y >= 0) and (1255 <= self.step_x <= 1260):
            return True
        return False

    def update(self,  dt):

        if self.finish():
            self.restart()
            return

        if 0 <= self.step_x <= self.wall.width - self.size and 0 <= self.step_y <= self.wall.height - self.size \
                and self.check_collision(self.step_x, self.step_y):
            self.x = self.step_x
            self.y = self.step_y

        self.get_distances()
        # обновляем список вершин квадрата с новыми координатами
        vertices = [
            self.x, self.y,
            self.x, self.y + self.size,
            self.x + self.size, self.y + self.size,
            self.x + self.size, self.y,
        ]
        self.vertex_list.vertices = vertices

    def get_distances(self, cube_size=20):
        distances = [0, 0, 0, 0]  # Расстояния в 4 направлениях: [вправо, влево, вверх, вниз]
        agent_x = (self.x + self.x + 10) / 2
        agent_y = (self.y * 2 + 10) / 2
        wall_matrix = self.wall.matrix

        # Получаем размеры лабиринта
        wall_height = len(wall_matrix)
        wall_width = len(wall_matrix[0])

        # Определяем координаты агента внутри клетки лабиринта
        agent_cell_x = int(agent_x // cube_size)
        agent_cell_y = wall_height - int(agent_y // cube_size) - 1
        print(agent_x, agent_cell_x, "<--x")
        print(agent_y, agent_cell_y, "<--y")
        # Вычисляем расстояние до стены вправо
        for col in range(agent_cell_x + 1, wall_width):
            if wall_matrix[agent_cell_y][col] == 0:  # Если стена найдена
                distances[0] = (col - agent_cell_x)-1  # Вычисляем расстояние до стены
                break

        # Вычисляем расстояние до стены влево
        for col in range(agent_cell_x - 1, 0, -1):
            if wall_matrix[agent_cell_y][col] == 0:  # Если стена найдена
                distances[1] = (agent_cell_x - col)-1  # Вычисляем расстояние до стены
                break
            else:
                distances[1] = agent_cell_x

        # Вычисляем расстояние до стены вниз
        for row in range(agent_cell_y + 1, wall_height):
            if wall_matrix[row][agent_cell_x] == 0:  # Если стена найдена
                distances[2] = (row - agent_cell_y)-1  # Вычисляем расстояние до стены
                break

        # Вычисляем расстояние до стены вверх
        for row in range(agent_cell_y - 1, -1, -1):
            if wall_matrix[row][agent_cell_x] == 0:  # Если стена найдена
                distances[3] = (agent_cell_y - row)-1  # Вычисляем расстояние до стены
                break
            else:
                distances[3] = agent_cell_y

        print(distances)
        return distances

