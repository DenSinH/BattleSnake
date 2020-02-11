from pprint import pprint


dirs = {
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up"
}

INFINITY = 9999999


def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class Path(object):

    def __init__(self, end, prevdir=None, path=None, firstdir=None):
        self.path = path or [end]
        self.end = end
        self.prevdir = prevdir
        self.firstdir = firstdir

    def __getitem__(self, item):
        return self.path[item]

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        return iter(self.path)

    def __repr__(self):
        return str(self.path)

    def __add__(self, other):
        assert other.path[0] == self.end
        assert len(other) > 1
        return Path(other.end, prevdir=other.prevdir, path=self.path + other.path[1:], firstdir=self.firstdir)

    def move(self, direction):
        prev = self.path[-1]
        extended = self.path + [(prev[0] + direction[0], prev[1] + direction[1])]
        return Path(extended[-1], direction, extended, self.firstdir or direction)

    def get(self):
        return dirs.get(self.firstdir)

    def first_head(self):
        if len(self.path) < 2:
            return None
        return self.path[1]

    def dist(self, point):
        return len(self) + manhattan(self.end, point)


class Snake(object):

    def __init__(self, body, health, **kwargs):
        self.health = health
        self.body = [(pos["x"], pos["y"]) for pos in body]
        self.head = self.body[0]

    def __len__(self):
        return len(self.body)

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
        for direction in dirs:
            # moving backwards is not allowed
            if start.prevdir == (-direction[0], -direction[1]):
                pass
            else:
                next_end = (start.end[0] + direction[0], start.end[1] + direction[1])

                # moving into borders or other snakes is not allowed
                if 0 <= next_end[0] < self.width and 0 <= next_end[1] < self.height:
                    for snake in self.snakes + [self.you]:
                        if next_end in snake.body:
                            break
                    else:
                        yield start.move(direction)

    def score(self, path):
        s = 0
        for spot in path:
            # we dont want to move our head next to a stronger snakes head
            if any(manhattan(path.first_head(), snake.head) == 1
                   and snake.strength() >= self.you.strength() for snake in self.snakes):
                return -INFINITY

            # snake likes to be next to game border if it is not next to another snake
            if any(spot in snake.body for snake in self.snakes):
                if spot[0] in [0, self.width - 1]:
                    s -= 3

                elif spot[1] in [0, self.height - 1]:
                    s -= 3

                else:
                    s += 3

            else:
                if spot[0] in [0, self.width - 1]:
                    s += 1

                elif spot[1] in [0, self.height - 1]:
                    s += 1

            # snake likes to be next to own body even more
            if spot in self.you.body:
                s += 5

            # snake likes to have options
            if len(path) > 1:
                s += 5 * sum([1 for i in self.flow(Path(path[1]))])

        return s

    def components(self, *extra):
        walls = {part for snake in self.snakes for part in snake.body} | set(self.you.body) | set(extra)
        components = []

        for x in range(self.width):
            for y in range(self.height):
                point = (x, y)
                if point in walls:
                    continue

                new_components = []
                current = {(x, y)}

                for component in components:
                    if any(manhattan(point, p) == 1 for p in component):
                        current |= component
                    else:
                        new_components.append(component)

                components = new_components + [current]

        return components

    def find_best(self, paths):
        """
        :param paths: Path[]
        :return: Path
        """

        return max(paths, key=self.score)

    def move(self):
        # todo: find longest path if in small connected component
        # no food case:
        if len(self.food) == 0:
            return max(self.flow(Path(self.you.head)), key=self.score).get()

        found = {self.you.head: 1}
        generation = {(self.you.head, None): [Path(self.you.head)]}
        next_generation = {}
        components = self.components()

        print("COMPONENTS BEFORE", [len(component) for component in components])

        # find best path among shortest paths
        shortest_paths = []

        # flow outward to find closest food and move along that path
        while generation:
            print(len(generation))

            for start, prevdir in generation:

                for extension in self.flow(Path(start, prevdir=prevdir)):

                    if extension.end in self.food:

                        # empty component
                        for component in components:
                            # todo: in small boards/late game, any component might be smaller than the snake
                            if extension.end in component and len(component) < len(self.you):
                                print(f"Did not allow path to {extension.end} because component too small")
                                break
                        else:
                            print("FOUND SHORTEST PATH TO", extension.end)
                            shortest_paths += [path + extension for path in generation[(start, prevdir)]]

                    if extension.end not in found:
                        found[extension.end] = len(generation[(start, prevdir)][0])
                    else:
                        for path in generation[(start, prevdir)]:
                            if len(path) + 1 <= found[extension.end]:
                                if (extension.end, extension.prevdir) not in next_generation:
                                    next_generation[(extension.end, extension.prevdir)] = []
                                next_generation[(extension.end, extension.prevdir)].append(path + extension)

            # only do best path if it is not dangerous
            if shortest_paths:
                for path in shortest_paths:
                    print(path.path)
                best = self.find_best(shortest_paths)
                if self.score(best) > -INFINITY:
                    return best.get()
                shortest_paths = []

            generation = next_generation
            next_generation = {}

        # todo: no path to food, choose largest area, largest path?
        choices = {}
        for next_path in self.flow(Path(self.you.head)):
            choices[next_path.get()] = 0

            # find largest area to go to
            for component in components:
                if next_path.end in component and len(component) < len(self.you):
                    choices[next_path.get()] += len(component) - len(self.you)

        return max(choices, key=lambda i: choices[i])


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


"""
todo: aggression (sectioning off other snakes, head to head colission)
"""
