import numpy as np
import random
import sys
import os

# fixing api import error
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from PriorityQueue import PriorityQueue

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

    def __lt__(self, other):
        assert isinstance(other, Path)
        return len(self) < len(other)

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        return iter(self.path)

    def __repr__(self):
        return str(self.path)

    def __add__(self, other):
        assert other.path[0] == self.end
        assert len(other) > 1
        return Path(other.end, prevdir=other.prevdir, path=self.path + other.path[1:],
                    firstdir=self.firstdir if len(self) > 1 else other.firstdir)

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
        self.food = {(pos["x"], pos["y"]) for pos in food}
        self.snakes = []
        for snake in snakes:
            if snake["id"] != you["id"]:
                self.snakes.append(Snake(**snake))

        self.you = Snake(**you)

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

    def leftover(self, component, path, end, target):
        to_check = {end}
        checked = set()
        path_set = set(path.path)

        # !!! leftover is not 0 because end is not in path !!!
        leftover = 1
        target_reached = False

        while to_check:
            current = to_check.pop()
            checked.add(current)

            for direction in dirs:
                nxt = (current[0] + direction[0], current[1] + direction[1])

                if nxt == target:
                    target_reached = True

                if nxt not in component:
                    continue

                if nxt in checked:
                    continue

                if nxt in path_set:
                    continue

                to_check.add(nxt)
                leftover += 1

        if target_reached:
            return leftover
        return -1

    def flow(self, generation, target, field):
        """
        :param generation: np.array([int, int])[]
        :param target:  set((int, int))
        :param field: np.2darray()
        :return: np.2darray, [(int, int)]

        flows distance outward from starting generation of points
        """

        dist = 1

        target_found = set()

        rows, cols = zip(*generation)
        field[rows, cols] = 0

        while generation and not target_found:

            generation_size = len(generation)
            for i in range(generation_size):
                current = generation.pop(0)

                for direction in dirs:
                    nxt = current + direction
                    if (nxt[0], nxt[1]) in target:
                        target_found.add((nxt[0], nxt[1]))
                        field[nxt[0], nxt[1]] = dist
                    elif np.all(nxt >= 0) and np.all(nxt < np.shape(field)):
                        if dist < field[nxt[0], nxt[1]]:
                            field[nxt[0], nxt[1]] = dist
                            generation.append(nxt)

            dist += 1

        return field, target_found

    def get_target(self, component):
        # determine target for longest path
        target = None
        target_score = INFINITY

        # find walls in components that are parts of snake
        for spot in component:
            for direction in dirs:
                nxt = (spot[0] + direction[0], spot[1] + direction[1])

                if nxt in component:
                    continue

                if not (0 <= nxt[0] < self.width and 0 <= nxt[1] <= self.height):
                    continue

                for snake in self.snakes + [self.you]:
                    for i in range(len(snake.body)):
                        if nxt == snake.body[i]:
                            nxt_score = len(snake) - i
                            # todo: check component connections?
                            break
                    else:
                        continue
                    break
                else:
                    continue

                if nxt_score < target_score or target is None:
                    target = nxt
                    target_score = nxt_score

        return target

    def longest_path(self, target, component):
        print("TARGET:", target)
        paths = PriorityQueue()
        paths.put(Path(self.you.head), 0)

        longest = Path(self.you.head)

        while not paths.empty():

            current = paths.get()

            if len(paths) > 100:
                print(len(paths))

            # todo: check if we can finish path
            # todo: stop if taking too long

            for direction in dirs:
                # moving backwards is not allowed
                if current.prevdir == (-direction[0], -direction[1]):
                    continue

                next_end = (current.end[0] + direction[0], current.end[1] + direction[1])

                # can't loop back in on itself
                if next_end in current.path:
                    continue

                if next_end == target:
                    longest = max(longest, current.move(direction), key=lambda p: len(p))
                    if len(longest) >= len(component) - 1:
                        print("LONGEST PATH FOUND EARLY:", longest.path)
                        return longest
                    continue

                # moving into borders or other snakes is not allowed (unless target)
                if 0 <= next_end[0] < self.width and 0 <= next_end[1] < self.height:

                    for snake in self.snakes + [self.you]:
                        if next_end in snake.body:
                            break
                    else:
                        leftover = self.leftover(component, current, next_end, target)
                        if leftover >= 0:
                            if len(current) + leftover >= len(longest):
                                paths.put(current.move(direction), len(current) + manhattan(next_end, target))

        if len(longest) == 1:
            # THIS SHOULD NEVER HAPPEN:
            for i in range(10):
                print("NO LONGEST PATH FOUND")
            return None

        print("LONGEST PATH IS", longest.path)
        return longest

    def score_spot(self, spot):

        s = 0

        for snake in self.snakes:
            if manhattan(spot, snake.head) == 1:
                if snake.strength() >= self.you.strength():
                    # other is likely to go there
                    if spot in self.food:
                        return -INFINITY ** 2
                    return -INFINITY
                else:
                    # slight bit of aggression
                    s += 10

        # snake likes to be next to game border if it is not next to another snake
        for snake in self.snakes:
            for part in snake.body[:-1]:
                if manhattan(spot, part) == 1:
                    next_over = (2 * spot[0] - part[0], 2 * spot[1] - part[1])

                    if next_over[0] in [-1, self.width]:
                        s -= 3

                    elif next_over[1] in [-1, self.height]:
                        s -= 3

                    elif any(next_over in _snake.body[:-1] for _snake in self.snakes):
                        s -= 3

                    else:
                        s += 3

                    break

        else:
            if spot[0] in [0, self.width - 1]:
                s -= 1

            elif spot[1] in [0, self.height - 1]:
                s -= 1

        # snake likes to be next to own body even more
        if any(manhattan(spot, part) == 1 for part in self.you.body if part != self.you.head):
            s += 5

        return s

    def score(self, path, score_field):
        # prefer to go straight slightly
        return sum(score_field[spot] for spot in path) + int(path.prevdir == (self.you.head[0] - self.you.body[1][0],
                                                                              self.you.head[1] - self.you.body[1][1]))

    def get_best(self, paths, allowed_squares):
        # assign score values to allowed_squares and calculate score
        score_field = np.zeros((self.width, self.height))

        for spot in zip(*np.nonzero(allowed_squares)[:2]):
            score_field[spot] = self.score_spot(spot)

        return max(paths, key=lambda p: self.score(p, score_field))

    def no_food(self, components, next_components):

        choices = {}
        comp_reached = {}

        for direction in dirs:
            nxt = (self.you.head[0] + direction[0], self.you.head[1] + direction[1])

            if not (0 <= nxt[0] < self.width and 0 <= nxt[1] <= self.height):
                continue

            if any(nxt in snake.body for snake in self.snakes + [self.you]):
                continue

            if any(manhattan(nxt, snake.head) == 1 and snake.strength() >= self.you.strength() for snake in
                   self.snakes):
                # other snake is likely to move to closest food
                multiplier = (self.width + 1) * (self.height + 1)
                multiplier -= min([manhattan(nxt, food) for food in self.food], default=0)

                # prefer to stay in larger area
                for component in components:
                    if nxt in component:
                        multiplier -= len(component)
                        comp_reached[dirs[direction]] = component
                        break

                choices[dirs[direction]] = - multiplier * INFINITY
                continue

            for component in components:
                if nxt in component:
                    choices[dirs[direction]] = 3 * len(component)
                    comp_reached[dirs[direction]] = component
                    break
            else:
                continue

            for next_component in next_components:
                if nxt in components:
                    choices[dirs[direction]] = 3 * len(next_component)
                    break

            choices[dirs[direction]] += self.score_spot(nxt)

        if len(choices) == 0:
            return random.choice(list(dirs.values()))

        # check what components are reached by the best directions
        best = max(choices, key=lambda d: choices[d])
        dirs_to_best = 0
        best_reached = []

        for d in choices:
            if choices[d] == choices[best]:
                dirs_to_best += 1
                if comp_reached[d] not in best_reached:
                    best_reached.append(comp_reached[d])

        if dirs_to_best == 1:
            # only one best option
            return best

        elif len(best_reached) == 1 and len(best_reached[0]) < len(self.you):

            print("CHECKING LONGEST PATH")
            component = best_reached.pop()
            target = self.get_target(component)

            if target is not None:
                longest = self.longest_path(target, component)
                if longest is not None:
                    return longest.get()
                else:
                    print("THIS SHOULD NEVER HAPPEN, longest IS None")

            else:
                print("THIS SHOULD NEVER HAPPEN, target IS None")

            return random.choice(list(dirs.values()))

        return best

    def move(self):
        components = self.components()

        semi_allowed_food = set(self.food)

        for component in components:
            if len(component) < len(self.you):
                semi_allowed_food -= component

        # prepare field
        inf = self.width * self.height + 1
        head_field = inf * np.ones((self.width, self.height))

        # snakes are likely to grab food in a straight line from them
        for food in self.food:
            for snake in self.snakes:
                if manhattan(food, snake.head) <= manhattan(food, self.you.head):
                    if food[0] == snake.head[0] and not any((food[0], i) in _snake.body
                                                            for _snake in self.snakes
                                                            for i in range(min(food[1], snake.head[1]) + 1,
                                                                           max(food[1], snake.head[1]))):

                        semi_allowed_food.discard(food)
                        head_field[food[0], min(food[1], snake.head[1]):max(food[1], snake.head[1]) + 1] = -1

                    elif food[1] == snake.head[1] and not any((i, food[1]) in _snake.body
                                                              for _snake in self.snakes
                                                              for i in range(min(food[0], snake.head[0]) + 1,
                                                                             max(food[0], snake.head[0]))):

                        semi_allowed_food.discard(food)
                        head_field[min(food[0], snake.head[0]):max(food[0], snake.head[0]) + 1, food[1]] = -1

        allowed_food = set(semi_allowed_food)

        # we don't like our snake to move across forbidden lines either
        next_components = self.components(*[(snake.head[0] + direction[0], snake.head[1] + direction[1])
                                            for direction in dirs for snake in self.snakes]
                                           + [tuple(p) for p in np.argwhere(head_field == -1)])

        for snake in self.snakes + [self.you]:
            rows, cols = zip(*snake.body)
            head_field[rows, cols] = -1

        # don't allow food that other snakes could cut off
        for next_component in next_components:

            if len(next_component) < len(self.you):

                for component in components:
                    if next_component & component:
                        if len(component) < len(self.you):
                            semi_allowed_food -= component
                    break

                rows, cols = zip(*next_component)
                head_field[rows, cols] = -1

                allowed_food -= next_component

        # no food case:
        if len(allowed_food) == 0:
            if len(semi_allowed_food) > 0 \
                    and 1.5 * min([manhattan(food, self.you.head) for food in semi_allowed_food]) >= self.you.health:
                print("LOW ENERGY, ALLOWED SOME FOOD")
                allowed_food = semi_allowed_food
            else:
                print("NO FOOD ALLOWED")
                return self.no_food(components, next_components)

        food_field = np.array(head_field)

        head_field, food_found = self.flow([np.array(self.you.head)], allowed_food, head_field)

        # no reachable food case
        if not food_found:
            print("NO PATH TO FOOD")
            return self.no_food(components, next_components)

        food_field, _ = self.flow([np.array(food) for food in food_found], {self.you.head}, food_field)

        allowed_squares = (head_field + food_field) == food_field[self.you.head]

        paths = [Path(self.you.head)]

        while paths and paths[0].end not in allowed_food:

            for i in range(len(paths)):
                current = paths.pop(0)

                for direction in dirs:
                    # moving backwards is not allowed
                    if current.prevdir == (-direction[0], -direction[1]):
                        continue

                    next_end = (current.end[0] + direction[0], current.end[1] + direction[1])

                    # only check if first move is next to other snake's head first generation
                    if current.end == self.you.head:
                        if any(manhattan(next_end, snake.head) == 1
                               and snake.strength() >= self.you.strength() for snake in self.snakes):
                            continue

                    # moving into borders or other snakes is not allowed
                    if 0 <= next_end[0] < self.width and 0 <= next_end[1] < self.height:

                        # have to move away from the heads original position
                        if head_field[next_end] > head_field[current.end] and food_field[next_end] < food_field[
                                current.end]:
                            for snake in self.snakes + [self.you]:
                                if next_end in snake.body:
                                    break
                            else:
                                paths.append(current.move(direction))

            # if there is only one path after at least 1 generation, then there is only once choice
            if len(paths) == 1:
                print("ONE CHOICE")
                return paths[0].get()

            # if there are too many allowed squares, just find the best move after one generation
            elif len(paths) > 150 and np.count_nonzero(allowed_squares) >= 30:
                print("TOO MANY POSSIBILITIES")
                return self.get_best(paths, allowed_squares).get()

        print(len(paths), "PATHS FOUND TO FOOD")
        print(np.count_nonzero(allowed_squares), "ALLOWED SQUARES")

        if len(paths) == 0:
            print("WON'T MOVE NEXT TO BETTER SNAKES HEAD")
            return self.no_food(components, next_components)

        best = self.get_best(paths, allowed_squares)
        return best.get()


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
