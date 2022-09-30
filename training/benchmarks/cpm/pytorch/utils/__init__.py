import inspect

from .check import check_config
from .dist import *

def is_property(value):
    status = [
        not callable(value),
        not inspect.isclass(value),
        not inspect.ismodule(value),
        not inspect.ismethod(value),
        not inspect.isfunction(value),
        not inspect.isbuiltin(value),
        "classmethod object" not in str(value)
    ]

    return all(status)