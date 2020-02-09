dirs = {
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up"
}


class Path(object):

    def __init__(self, end, prevdir=None, path=None, firstdir=None):
        self.path = path or [end]
        self.end = end
        self.prevdir = prevdir
        self.firstdir = firstdir

    def __getitem__(self, item):
        return self.path[item]

    def move(self, direction):
        prev = self.path[-1]
        extended = self.path + [(prev[0] + direction[0], prev[1] + direction[1])]
        return Path(extended[-1], direction, extended, self.firstdir or direction)

    def get(self):
        return dirs[self.firstdir]


class Snake(object):

    def __init__(self, body, health, **kwargs):
        self.health = health
        self.body = [(pos["x"], pos["y"]) for pos in body]
        self.head = self.body[0]

    def strength(self):
        return len(self.body)


class Game(object):

    def __init__(self, height, width, food, snakes, you, **kwargs):
        self.height = height
        self.width = width
        self.food = [(pos["x"], pos["y"]) for pos in food]
        self.snakes = []
        for snake in snakes:
            if snake["id"] != you["id"]:
                self.snakes.append(Snake(**snake))

        self.you = Snake(**you)

    def flow(self, start):
        result = []
        for direction in dirs:
            # moving backwards is not allowed
            if start.prevdir == (-direction[0], -direction[1]):
                pass
            else:
                next_path = start.move(direction)
                # moving into borders or other snakes is not allowed
                if 0 <= next_path.end[0] < self.width and 0 <= next_path.end[1] < self.height:
                    for snake in self.snakes + [self.you]:
                        if next_path.end in snake.body:
                            break
                    else:
                        result.append(next_path)

        return result

    def move(self):
        found = set(self.you.head)
        todo = [Path(self.you.head)]

        # flow outward to find closest food and move along that path
        while todo:
            current = todo.pop(0)
            for next_path in self.flow(current):
                if next_path.firstdir is not None and not len(self.food):
                    return next_path.get()

                if next_path.end in self.food:
                    return next_path.get()

                if next_path.end not in found:
                    found.add(next_path.end)
                    todo.append(next_path)

            # todo: no path to food, choose largest area?
            if len(todo) == 0:
                return current.get()


def make_move(data):
    game = Game(you=data["you"], **data["board"])
    return game.move()


"""
{'board': {'food': [{'x': 0, 'y': 2},
                    {'x': 18, 'y': 2},
                    {'x': 18, 'y': 9},
                    {'x': 4, 'y': 16}],
           'height': 19,
           'snakes': [{'body': [{'x': 17, 'y': 8},
                                {'x': 17, 'y': 9},
                                {'x': 17, 'y': 10}],
                       'health': 91,
                       'id': 'gs_63XBjDGkmVgKtkrvPr7bVCrH',
                       'name': 'matthewlehner / Ouroboros'},
                      {'body': [{'x': 1, 'y': 18},
                                {'x': 1, 'y': 17},
                                {'x': 1, 'y': 16}],
                       'health': 91,
                       'id': 'gs_RFGgR4bqkRm9R3g3mTHRKqYV',
                       'name': 'densinh / Dennis'}],
           'width': 19},
 'game': {'id': 'bd82b19c-1477-4dc6-bd33-ebe5e5e145ec'},
 'turn': 9,
 'you': {'body': [{'x': 1, 'y': 18}, {'x': 1, 'y': 17}, {'x': 1, 'y': 16}],
         'health': 91,
         'id': 'gs_RFGgR4bqkRm9R3g3mTHRKqYV',
         'name': 'densinh / Dennis'}}
"""


"""
x = 0, y = 0 top left

x = l - 1, y = l - 1 bottom right
first element is head
"""
