import inspect

from .check import check_config


def is_property(value):
    status = [
        not callable(value),
        not inspect.isclass(value),
        not inspect.ismodule(value),
        not inspect.ismethod(value),
        not inspect.isfunction(value),
        not inspect.isbuiltin(value),
        not isinstance(value, classmethod),
    ]

    return all(status)
