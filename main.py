import pyglet
import numpy


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

    def update(self, dt):

        # изменяем координаты квадрата на заданный вектор
        dx = (self.keys[pyglet.window.key.RIGHT] - self.keys[pyglet.window.key.LEFT]) * 1
        dy = (self.keys[pyglet.window.key.UP] - self.keys[pyglet.window.key.DOWN]) * 1

        x = self.x + dx
        y = self.y + dy

        # for w in wall:
        #     if w.x <= x <= w.x:
        if 0 <= x <= width - self.size:
            self.x += dx
        if 0 <= y <= height - self.size:
            self.y += dy

        # обновляем список вершин квадрата с новыми координатами
        vertices = [
            self.x, self.y,
            self.x, self.y + self.size,
            self.x + self.size, self.y + self.size,
            self.x + self.size, self.y,
        ]
        self.vertex_list.vertices = vertices


class Wall:
    def __init__(self, x, y, w, h, size_pass, pos):
        if pos == 'h':
            wall1 = [
                x, y,
                x, y + h,
                x + w, y + h,
                x + w, y,
            ]
            wall2 = [
                x, y + h + size_pass,
                x, y + 2 * h + size_pass,
                x + w, y + 2 * h + size_pass,
                x + w, y + h + size_pass,
            ]
        else:
            wall1 = [
                x, y,
                x, y + w,
                x + h, y + w,
                x + h, y,
            ]
            wall2 = [
                x + h + size_pass, y,
                x + h + size_pass, y + w,
                x + 2 * h + size_pass, y + w,
                x + 2 * h + size_pass, y,
            ]

        # создаем объект вершинного буфера
        self.vertex_list1 = pyglet.graphics.vertex_list(4, ('v2i', wall1))
        self.vertex_list2 = pyglet.graphics.vertex_list(4, ('v2i', wall2))
        # сохраняем начальные координаты квадрата
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.size_pass = size_pass
        self.pos = pos
        self.color = (255, 255, 255)
        self.wall1 = wall1
        self.wall2 = wall2

    def draw(self):
        pyglet.gl.glColor3f(*self.color)
        self.vertex_list1.draw(pyglet.gl.GL_QUADS)
        self.vertex_list2.draw(pyglet.gl.GL_QUADS)


# создаем нужные рамки для окна

width = 1280
height = 720

# создаем окно
window = pyglet.window.Window(width, height)

pos_x = 10
pos_y = 10
size_xy = 10
# создаем объект квадрата
people = Human(pos_x, pos_y, size_xy)
# создаем объект стены
wall = numpy.empty(2, dtype=object)
wall[0] = Wall(75, 130, 400, 5, 20, 'w')
wall[1] = Wall(100, 100, 400, 5, 20, 'h')


# функция, которая вызывается каждый кадр
@window.event
def on_draw():
    window.clear()
    people.draw()
    for i in wall:
        i.draw()


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
