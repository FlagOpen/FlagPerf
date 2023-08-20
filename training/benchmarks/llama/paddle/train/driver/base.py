# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import inspect
from typing import Iterable

from . import log_event, mod_util, perf_logger, config_manager
from .event import Event, EventHandleRecord, EventManager
from icecream import ic

class Driver(object):

    def __init__(self, config, mutable_params):
        self.config = config
        self.mutable_params = mutable_params
        self.is_distributed = False
        self.event_handlers = dict()
        self.extern_modules = dict()
        self.logger = None

    def setup_config(self, parser):
        parser.add_argument(
            "--extern_module_dir",
            type=str,
            required=False,
            help="The full path to the root of external modules")
        parser.add_argument(
            "--extern_config_dir",
            type=str,
            required=False,
            help="Specifies the directory of the external config files")
        parser.add_argument("--extern_config_file",
                            type=str,
                            required=False,
                            help="The external config file to use")
        parser.add_argument(
            "--enable_extern_config",
            action="store_true",
            help="Sets True if external config parameters are allowd")
        parser.add_argument("--data_dir",
                            type=str,
                            default="/mnt/dataset/",
                            help="Data directory.")
        parser.add_argument("--vendor",
                            type=str,
                            required=True, # 改true
                            help="The accelerator vendor that run the located.")
        known_args, unknown_args = parser.parse_known_args()

        config_manager.activate(self.config, self.mutable_params,
                                known_args.extern_config_dir,
                                known_args.extern_config_file,
                                known_args.enable_extern_config, known_args, unknown_args)

        if known_args.extern_module_dir:
            mod_util.install_extern_modules(known_args.extern_module_dir,
                                            self.extern_modules)
        self.logger = perf_logger.PerfLogger.get_default_logger(
            rank=self.config.local_rank)
        
        # consider different config format between framework，e.g. pytorch & tensorflow
        try:
            log_freq = self.config.log_freq
        except AttributeError:
            log_freq = self.config.train.time_history.log_steps
        event_manager = log_event.LogEventManager(self.logger,
                                                  log_freq=log_freq)
                                                  
        event_manager.register_event_handlers(self)
        for _, mod in self.extern_modules.items():
            for cls in mod_util.find_derived_classes(EventManager, mod):
                event_manager = cls()
                event_manager.register_event_handlers(self)

    def setup_modules(self, *args):
        for arg in args:
            if inspect.ismodule(arg):
                print(str(arg) + " replace by " + str(self.extern_modules))
                mod_util.replace_submodules(arg, self.extern_modules)
            elif isinstance(arg, dict):
                print(str(arg) + " remap by " + str(self.extern_modules))
                mod_util.remap_modules(arg, self.extern_modules)
            elif isinstance(arg, object):  # TODO
                pass
            else:
                raise TypeError('Can either be a module or a dict')

    def launch(self):
        self.event(Event.LAUNCH_TRAINING)
        config_path: str = self.config.config
        config_dict = self.config.get_properties_from_config(self.config)
        for key, value in config_dict.items():
            if type(value) not in [int, float, str, bool
                                   ] and not isinstance(value, Iterable):
                config_dict[key] = str(value)

        # Like /path/to/vendor/model-framework/config/config_xxx.py
        if config_path.startswith("."):
            config_path = os.path.abspath(config_path)

        config_path_nodes = config_path.rsplit(sep="/", maxsplit=4)
        vendor = config_path_nodes[1]
        model = config_path_nodes[2]
        self.logger.init_logger(vendor=vendor,
                                model=model,
                                config_path=config_path,
                                config=config_dict,
                                stacklevel=log_event.STACKLEVEL)

    def register_event_handler(self, ehr: EventHandleRecord):
        e = ehr.event
        if not e in self.event_handlers:
            self.event_handlers[e] = []
        self.event_handlers[e].append(ehr)

    def event(self, e: Event, *args, **kwargs):
        assert e in self.event_handlers
        for h in self.event_handlers[e]:
            h.handle(*args, **kwargs)
