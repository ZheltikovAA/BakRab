import pyglet
import random
import labirint

class Human:
    def __init__(self, x, y, size):
        # создаем список вершин квадрата
        vertices = [
            x, y,
            x, y + size,
               x + size, y + size,
               x + size, y,
        ]
        # создаем объект вершинного буфера
        self.vertex_list = pyglet.graphics.vertex_list(4, ('v2i', vertices))

        # сохраняем начальные координаты квадрата
        self.x = x
        self.y = y
        self.size = size
        self.vertices = vertices
        # устанавливаем начальное состояние клавиш
        self.keys = {
            pyglet.window.key.LEFT: False,
            pyglet.window.key.RIGHT: False,
            pyglet.window.key.UP: False,
            pyglet.window.key.DOWN: False,
        }

        self.color = (255, 0, 0)

    def draw(self):
        pyglet.gl.glColor3f(*self.color)
        self.vertex_list.draw(pyglet.gl.GL_QUADS)

    def check_collision(self, x, y):
        # Проверяем столкновение с каждой стеной
        for i, row in enumerate(wall.matrix):
            for j, value in enumerate(row):
                if value == 1:
                    wall_x = j * wall.cell_width
                    wall_y = (wall.matrix_height - i - 1) * wall.cell_height
                    if x + self.size > wall_x and x < wall_x + wall.cell_width and \
                            y + self.size > wall_y and y < wall_y + wall.cell_height:
                        return True
        return False
    # def maze_collision_check(self, x_, y_):
    #     maze_cell_x = x_ // len(wall.matrix[0])
    #     maze_cell_y = y_ // len(wall.matrix)
    #     return wall.matrix[maze_cell_y][maze_cell_x] != 0
    def update(self, dt):

        # изменяем координаты квадрата на заданный вектор
        dx = (self.keys[pyglet.window.key.RIGHT] - self.keys[pyglet.window.key.LEFT]) * 10
        dy = (self.keys[pyglet.window.key.UP] - self.keys[pyglet.window.key.DOWN]) * 10

        x = self.x + dx
        y = self.y + dy

        if 0 <= x <= width - self.size and 0 <= y <= height - self.size and self.check_collision(x, y):
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



class Wall:
    def __init__(self, width_lab, height_lab):
        matrix = [[0] * width_lab for _ in range(height_lab)]
        self.matrix = labirint.labirint(matrix, width_lab, height_lab)
        self.matrix_height = height_lab
        self.matrix_width = width_lab
        # Размеры ячейки
        self.cell_width = window.width // len(self.matrix[0])
        self.cell_height = window.height // len(self.matrix)

    def draw(self):
        batch = pyglet.graphics.Batch()

        for i, row in enumerate(self.matrix):
            for j, value in enumerate(row):
                if value == 1:
                    # Отрисовка заполненной ячейки
                    x = j * self.cell_width
                    y = (len(self.matrix) - i - 1) * self.cell_height  # Инвертируем направление отрисовки по y для правильного расположения
                    vertex_list = batch.add(4, pyglet.gl.GL_QUADS, None,
                                            ('v2i',
                                             [x, y, x, y + self.cell_height, x + self.cell_width, y + self.cell_height, x + self.cell_width,
                                              y]),
                                            ('c3B', [255, 255, 255] * 4))  # Цвет заполненной ячейки (белый)

        batch.draw()

# создаем нужные рамки для окна

width = 1280
height = 720

# создаем окно
window = pyglet.window.Window(width, height)

pos_x = 0
pos_y = 20
size_xy = 10
# создаем объект квадрата
people = Human(pos_x, pos_y, size_xy)
wall = Wall(64, 36)


# функция, которая вызывается каждый кадр
@window.event
def on_draw():
    window.clear()
    wall.draw()
    people.draw()


# функция, которая вызывается при нажатии клавиши
@window.event
def on_key_press(symbol, modifiers):
    if symbol in people.keys:
        people.keys[symbol] = True


# функция, которая вызывается при отпускании клавиши
@window.event
def on_key_release(symbol, modifiers):
    if symbol in people.keys:
        people.keys[symbol] = False


# функция, которая вызывается каждый кадр
def update(dt):
    people.update(dt)


# устанавливаем интервал обновления
pyglet.clock.schedule_interval(update, 1 / 165)

pyglet.app.run()
