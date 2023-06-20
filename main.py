import pyglet
import Human
import Wall


class Game_maze(pyglet.window.Window):
    def __init__(self, width=1280, height=720):
        super().__init__(width, height)
        self.width = width
        self.height = height
        self.start_pos_x = 0
        self.start_pos_y = 20
        self.size_xy = 10
        self.score = 0
        self.wall = Wall.Wall(64, 36, width, height)
        self.people = Human.Human(self.start_pos_x, self.start_pos_y, self.size_xy, self.wall)
        self.window = pyglet.window.Window(self.width, self.height)
    def update(self, dt):
        pyglet.clock.schedule_once(self.update, 0)
        self.people.update(dt)

    def on_draw(self):
        self.window.clear()
        self.wall.draw(self.people.score)
        self.people.draw()

    # функция, которая вызывается при нажатии клавиши

    def on_key_press(self, symbol, modifiers):
        pyglet.clock.schedule_once(game.update, 0)
        if symbol in self.people.keys:
            self.people.keys[symbol] = True

    # функция, которая вызывается при отпускании клавиши

    def on_key_release(self, symbol, modifiers):
        if symbol in self.people.keys:
            self.people.keys[symbol] = False

    def start_game(self):
        # pyglet.clock.schedule(self.update)
        pyglet.app.run()


if __name__ == "__main__":
    game = Game_maze()
    game.start_game()


