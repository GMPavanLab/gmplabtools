import json
from types import SimpleNamespace


def get_config(filename, section):
    return SimpleNamespace(**json.load(open(filename, 'r'))[section])
