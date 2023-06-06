import pyglet
# import NNConstructMaze
import Human
import Wall
import numba


width = 1280
height = 720

# создаем окно
window = pyglet.window.Window(width, height)

pos_x = 0
pos_y = 20
size_xy = 10
# создаем объект квадрата
wall = Wall.Wall(64, 36, width, height)
people = Human.Human(pos_x, pos_y, size_xy, wall)


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
