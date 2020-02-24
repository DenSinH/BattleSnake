import heapq


class PriorityQueue:
    """
    adapted from: https://www.redblobgames.com/pathfinding/a-star/implementation.html
    priority queue with highest priority first
    """

    def __init__(self):
        self.elements = []

    def __len__(self):
        return len(self.elements)

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (-priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]
