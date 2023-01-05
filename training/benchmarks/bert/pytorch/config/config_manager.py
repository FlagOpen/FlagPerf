import copy
import importlib
import inspect
import os
import sys
from argparse import ArgumentParser
from typing import Iterable, Mapping

import config as global_config
from . import _base as base_config
from .mutable_params import mutable_params

mutable_params = copy.copy(mutable_params)
immutable_params = set(global_config.__dict__.keys()) - set(mutable_params)


def get_config(config_path: str):
    if os.path.exists(config_path):
        abs_path = config_path
        sys.path.append(os.path.dirname(abs_path))
        config_path = os.path.basename(config_path).replace(".py", "")
        try:
            module = importlib.import_module(config_path)
        except Exception as ex:
            sys.path.pop(-1)
            raise ex
        sys.path.pop(-1)
    else:
        raise FileNotFoundError("Not found config:", config_path)

    return module


def get_annotations(other_modules: list=None):
    annotations = dict()

    if "__annotations__" in base_config.__dict__:
        annotations.update(base_config.__dict__["__annotations__"])

    if other_modules is not None:
        for mod in other_modules:
            if isinstance(mod, str):
                mod = get_config(mod)
            if "__annotations__" in mod.__dict__:
                annotations.update(mod.__dict__["__annotations__"])

    return annotations


def is_property(name: str, value):
    status = [
        not name.startswith('__'),
        not callable(value),
        not inspect.isclass(value),
        not inspect.ismodule(value),
        not inspect.ismethod(value),
        not inspect.isfunction(value),
        not inspect.isbuiltin(value),
    ]

    return all(status)


def get_properties_from_config(config):
    if not isinstance(config, Mapping):
        config = config.__dict__
    properties = dict()
    for name, value in config.items():
        if is_property(name, value):
            properties[name] = value

    return properties


def add_to_argparser(config: dict, parser: ArgumentParser, other_modules: list=None):
    annotations = get_annotations(other_modules)

    def get_property_type(name, value):
        if value is not None:
            return type(value)
        if name in annotations:
            return annotations[name]
        return str

    def add_args(parser, name, value, prefix=''):
        dtype = get_property_type(prefix + name, value)

        if dtype == str:
            parser.add_argument('--' + prefix + name, type=str, default=None)
        elif dtype == int:
            parser.add_argument('--' + prefix + name, type=int, default=None)
        elif dtype == float:
            parser.add_argument('--' + prefix + name, type=float, default=None)
        elif dtype == bool:
            parser.add_argument('--' + prefix + name, action=f"store_{str(not value).lower()}", default=None)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                add_args(parser, k, v, prefix=prefix + name + ".")
        elif isinstance(value, Iterable) and not isinstance(value, Mapping):
            parser.add_argument('--' + prefix + name, type=type(value[0]), nargs='+', default=None)
        # else:
        #     print(f'WARN: Cannot parse key {prefix + name} of type {type(value)}.')

    for name, value in config.items():
        if not is_property(name, value):
            continue
        add_args(parser, name, value)


def _merge_dict_to_config(src: dict, dist: dict, ignore_none=True):
    for arg, value in src.items():
        if ignore_none and value is None:
            continue
        dist[arg] = value


def parse_from_args(config: dict, parser=None, other_modules: list=None, with_config_env_name: bool=False):
    if parser is None:
        parser = ArgumentParser()

    add_to_argparser(config, parser, other_modules)
    if with_config_env_name:
        parser.add_argument("config", type=str, help="Config name")

    args = parser.parse_args()
    return args


def activate_config_env(name=None, parser=None, parse_args=True, with_config_env_name: bool=False):
    global_config_copy_ = copy.copy(global_config.__dict__)

    if parse_args:
        args_dict = dict()
        for mutable_param in mutable_params:
            args_dict[mutable_param] = global_config_copy_[mutable_param]
        args = parse_from_args(args_dict, parser, with_config_env_name=with_config_env_name)
        del args_dict
        if name is None and with_config_env_name:
            name = args.config

    if name is None:
        raise RuntimeError("Argument `name` must be given.")

    external_module_params = get_config(name).__dict__
    for immutable_param in immutable_params:
        if immutable_param in external_module_params:
            external_module_params.pop(immutable_param)

    _merge_dict_to_config(global_config_copy_, global_config.__dict__)
    _merge_dict_to_config(external_module_params, global_config.__dict__)
    if parse_args:
        _merge_dict_to_config(args.__dict__, global_config.__dict__)


def print_config(config=None):
    if config is None:
        config = global_config
    properties = get_properties_from_config(config)
    config_fields = []
    for name, value in properties.items():
        config_fields.append(f"{name}={value}")

    config_fields = ", ".join(config_fields)
    config_str = f"Config({config_fields})"
    print(config_str)