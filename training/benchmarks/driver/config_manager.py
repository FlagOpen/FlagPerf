# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import importlib
import inspect
import os
import sys
from argparse import ArgumentParser
from typing import Iterable, Mapping


def import_config(config_path: str):
    if os.path.exists(config_path):
        abs_path = config_path
        sys.path.append(os.path.dirname(abs_path))
        config_module = os.path.basename(config_path).replace(".py", "")
        try:
            module = importlib.import_module(config_module)
        except Exception as ex:
            raise ex
        finally:
            sys.path.pop(-1)

        return module
    else:
        raise f'{config_path} does not exist.'


def is_property(name: str, value):
    return all([
        not name.startswith('__'),
        not callable(value),
        not inspect.isclass(value),
        not inspect.ismodule(value),
        not inspect.ismethod(value),
        not inspect.isfunction(value),
        not inspect.isbuiltin(value),
    ])


def get_properties_from_config(config):
    if not isinstance(config, Mapping):
        config = config.__dict__
    properties = dict()
    for name, value in config.items():
        if is_property(name, value):
            properties[name] = value

    return properties


def add_to_argparser(config: dict, parser: ArgumentParser):

    def get_property_type(name, value):
        if value is not None:
            return type(value)
        return str

    def add_args(parser, name, value, prefix=''):
        dtype = get_property_type(prefix + name, value)

        if dtype in (str, int, float):
            parser.add_argument('--' + prefix + name, type=dtype, default=None)
        elif dtype == bool:
            parser.add_argument('--' + prefix + name,
                                action=f"store_{str(not value).lower()}",
                                default=None)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                add_args(parser, k, v, prefix=prefix + name + ".")
        elif isinstance(value, Iterable) and not isinstance(value, Mapping):
            parser.add_argument('--' + prefix + name,
                                type=type(value[0]),
                                nargs='+',
                                default=None)
        # else:
        #     print(f'WARN: Cannot parse key {prefix + name} of type {type(value)}.')

    for name, value in config.items():
        if not is_property(name, value):
            continue
        add_args(parser, name, value)


def _merge_dict_to_config(src: dict, target: dict, ignore_none=True):
    for arg, value in src.items():
        if ignore_none and value is None:
            continue
        target[arg] = value


def merge_args_to_extern_config(mutable_params, args):
    # TODO: A better config method, yaml or .ini?
    path =  os.path.join(args.extern_config_dir, args.extern_config_file)
    config_mod = import_config(path)
    for name, value in get_properties_from_config(config_mod).items():
        if name not in mutable_params:
            if not args.enable_extern_config:
                continue
            print(f"SET [Unknown or immutable] CONFIG {name} = {value}")
        else:
            print(f"SET CONFIG {name} = {value}")
        if args.__dict__.get(name) is None:
            setattr(args, name, value)
    return vars(args)

def activate(base_config,
             mutable_params,
             args):
    """ Find and activates externally defined config parameters by
    adding or updating attributes in the base config module.

    Args:
        path: specifies the directory path of the config file
        config_file: the config file name
        cmd_args: [] of extra arg strings

    Returns:
        mod: the enclosing module's namespace
    """
    path = args.extern_config_dir
    config_file = args.extern_config_file
    mutable_params_dict = dict()
    for mutable_param in mutable_params:
        mutable_params_dict[mutable_param] = getattr(base_config, mutable_param)

    if not path or not config_file:
        raise "path or config file's location was not specified."

    ext_config = os.path.join(os.path.abspath(path), config_file)
    parsed_params = merge_args_to_extern_config(mutable_params_dict, args)

    # TODO：后续考虑换一个更优雅的方式
    if "tensorflow2" in base_config.__path__:
        base_config.override(parsed_params.__dict__, False)
    else:
        _merge_dict_to_config(parsed_params, base_config.__dict__)

    if ext_config:
        config_path = ext_config
    else:
        config_path = base_config.__file__
    base_config.__dict__['config'] = config_path
