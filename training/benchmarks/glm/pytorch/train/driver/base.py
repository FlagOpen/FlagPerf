import os
import inspect
from functools import wraps
from typing import Iterable, List

import config

from . import log_event, mod_util, perf_logger
from .event import Event, EventHandleRecord, EventManager


class Driver(object):
    def __init__(self):
        self.config = config
        self.is_distributed = False
        self.event_handlers = dict()
        self.extern_modules = dict()

    def setup_config(self, parser):
        parser.add_argument("--extern_module_dir", type=str, required=False,
                            help="The full path to the root of external modules")
        parser.add_argument("--extern_config_dir", type=str, required=False,
                            help="Specifies the directory of the external config files")
        parser.add_argument("--extern_config_file", type=str,
                            required=False, help="The external config file to use")
        parser.add_argument("--enable_extern_config", action="store_true",
                            help="Sets True if external config parameters are allowd")
        path, args = parser.parse_known_args()
        config.activate(path.extern_config_dir,
                        path.extern_config_file, path.enable_extern_config, args)
        if path.extern_module_dir:
            mod_util.install_extern_modules(
                path.extern_module_dir, self.extern_modules)
        self.logger = perf_logger.PerfLogger.get_default_logger(
            rank=config.local_rank)
        event_manager = log_event.LogEventManager(
            self.logger, log_freq=config.log_freq)
        event_manager.register_event_handlers(self)
        for _, mod in self.extern_modules.items():
            for cls in mod_util.find_derived_classes(EventManager, mod):
                event_manager = cls()
                event_manager.register_event_handlers(self)

    def setup_modules(self, *args):
        for arg in args:
            if inspect.ismodule(arg):
                mod_util.replace_submodules(arg, self.extern_modules)
            elif isinstance(arg, dict):
                mod_util.remap_modules(arg, self.extern_modules)
            else:
                raise TypeError('Can either be a module or a dict')

    def launch(self):
        self.event(Event.LAUNCH_TRAINING)
        config_path: str = self.config.config
        config_dict = config.get_properties_from_config(self.config)
        for key, value in config_dict.items():
            if type(value) not in [int, float, str, bool] and not isinstance(value, Iterable):
                config_dict[key] = str(value)

        # Like /path/to/proj/submitter/model/config/config_xxx.py
        if config_path.startswith("."):
            config_path = os.path.abspath(config_path)

        config_path_nodes = config_path.rsplit(sep="/", maxsplit=4)
        submitter = config_path_nodes[1]
        model = config_path_nodes[2]
        self.logger.init_logger(submitter=submitter,
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
