import json
from types import SimpleNamespace

import toml


class WrongConfigFormat(Exception):
    pass


def get_config(filename: str, section: str) -> SimpleNamespace:
    with open(filename, 'r') as f:
        if filename.endswith('.json'):
            _config = json.load(f)
        elif filename.endswith('.toml'):
            _config = toml.load(f)
        else:
            raise WrongConfigFormat(
                f"Format of the '{filename}' is not supported. "
                "Available formats: .json, .toml"
            )
        return SimpleNamespace(**_config[section])
