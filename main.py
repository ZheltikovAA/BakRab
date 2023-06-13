import pyglet
# import NNConstructMaze
import Human
import Wall
import numba


def update(dt):
    people.update(dt)


width = 1280
height = 720
pos_x = 10
pos_y = 20
size_xy = 10

window = pyglet.window.Window(width, height)

wall = Wall.Wall(64, 36, width, height)
people = Human.Human(pos_x, pos_y, size_xy, wall)


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


pyglet.clock.schedule_interval(update, 1 / 720)
pyglet.app.run()

