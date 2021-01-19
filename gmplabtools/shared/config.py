import json, importlib, pkgutil, shutil, os
from types import SimpleNamespace
import toml


class WrongConfigFormat(Exception):
    pass


def _module_exists(module):
    try:
        return pkgutil.get_loader(module) is not None
    except ImportError:
        return False


def _importer(module_name, obj):
    if _module_exists(module_name):
        module = importlib.import_module(module_name)
        if hasattr(module, obj):
            return getattr(module, obj)
    elif os.path.isfile(module_name):
        base_module_name = os.path.basename(module_name).replace(".py", "")
        shutil.copy(module_name, ".")
        module = importlib.import_module(base_module_name)
        if hasattr(module, obj):
            return getattr(module, obj)


def import_external(config, class_name):
    try:
        module = importlib.import_module(".processing", package="gmplabtools.shared")
        return getattr(module, class_name)
    except AttributeError:
        if config.pymodule:
            class_type = _importer(config.pymodule, class_name)
        if config.mymodule and class_type is None:
            class_type = _importer(config.mymodule, class_name)
        if class_type is not None:
            return class_type
        else:
            msg = (
                f"Cannot find class {class_name} in system packages or in those provided with "
                f"in mymodule and pymodule config values."
            )
            raise ModuleNotFoundError(msg)


def get_config(filename, section=None):
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
        if section is not None:
            parse = _config[section]
        else:
            parse = _config
        return SimpleNamespace(**parse)
