import json
import random
import bottle
import os
import sys
import time

# fixing api import error
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import snake
from api import ping_response, start_response, move_response, end_response


@bottle.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), "snake.py"), "r") as f:
        return "<xmp>" + f.read() + "</xmp>"


@bottle.route('/static/<path:path>')
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.

    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root='static/')


@bottle.post('/ping')
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()


@bottle.post('/start')
def start():
    data = bottle.request.json

    """
    TODO: If you intend to have a stateful snake AI,
            initialize your snake state here using the
            request's data if necessary.
    """
    print("STARTING GAME")

    color = "#192821"

    return start_response(color)


@bottle.post('/move')
def move():
    data = bottle.request.json

    """
    choose a direction to move in.
    """

    t0 = time.time()
    direction = snake.make_move(data)
    t = time.time() - t0

    print("moving", direction, "in turn", data["turn"])
    print("decision made in", int(t * 1000), "milliseconds")

    if t > 0.45:
        for i in range(5):
            print("WARNING, TOO SLOW")

    return move_response(direction) or "up"


@bottle.post('/end')
def end():
    data = bottle.request.json

    """
    TODO: If your snake AI was stateful,
        clean up any stateful objects here.
    """
    print("GAME END")

    return end_response()


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug=os.getenv('DEBUG', True)
    )
