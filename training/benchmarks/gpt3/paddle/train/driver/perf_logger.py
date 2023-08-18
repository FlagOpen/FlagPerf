import os
import time
import logging
import json
from logging import currentframe
from typing import Union, Tuple, Optional
from collections import OrderedDict

from enum import IntEnum
from .event import Event

_srcfile = os.path.normcase(logging.addLevelName.__code__.co_filename)
from icecream import ic

class LogMeta:
    default_logger_name = "FlagPerfLogger"

    # Log format
    log_header = "PerfLog"
    log_template = "[{header}] {message}"


class LogKeys:
    # Submitted info
    submmiter: str = "submmiter"
    model: str = "model"
    optimizer_type: str = "optimizer_type"
    config: str = "config"
    config_path: str = "config_path"

    # Event
    event: str = "event"
    value: str = "value"

    # Training info
    step: str = "step"
    epoch: str = "epoch"
    loss: str = "loss"

    # Metadata
    metadata: str = "metadata"
    called_log_file = "file"
    called_log_file_lineno = "lineno"
    time_ms = "time_ms"
    rank = "rank"

    # Other message
    other_message: str = "other"


class LogLevel(IntEnum):

    INFO = 100
    SUBMITTION = 101

    @staticmethod
    def from_string(level: str):
        return LogLevel.__dict__[level.upper()]

    @classmethod
    def register_to_logging(cls, logging):
        for _, level in LogLevel.__members__.items():
            logging.addLevelName(level.value, level.name)


LogLevel.register_to_logging(logging)


class PerfLogger:

    _singleton = None

    def __init__(self,
                 rank: int,
                 level: LogLevel = LogLevel.SUBMITTION,
                 logger: logging.Logger = None):
        self.rank = rank
        self.level = level
        self.logger = logger or logging.Logger(LogMeta.default_logger_name)
        self.previous_log_time = None

    @property
    def _current_time_ms(self):
        current = int(time.time() * 1e3)
        self.previous_log_time = current
        return current

    def init_logger(self, submitter: str, model: str, config_path: str,
                    config: dict, *args, **kwargs):
        message = {
            LogKeys.submmiter: submitter,
            LogKeys.model: model,
            LogKeys.config_path: config_path,
            LogKeys.config: config
        }

        self.log(Event.SUBMIT_INFO, message, *args, **kwargs)

    def log(self,
            event: Event,
            level=None,
            rank=-1,
            message: Optional[Union[str, dict]] = None,
            *args,
            **kwargs):
        level = level or self.level
        show_log = any([rank == self.rank, rank == -1])
        if not show_log:
            return

        stacklevel = 1
        if "stacklevel" in kwargs:
            stacklevel = kwargs.pop("stacklevel")

        call_info = self.get_caller(stacklevel=stacklevel)

        message = self._encode_message(event, message, call_info, *args,
                                       **kwargs)
        self.logger.log(self.level.value, message)

    def _encode_message(self, event: Event, message: Union[str, dict],
                        call_info: Tuple[str, int], *args, **kwargs) -> str:
        if isinstance(message, str):
            message = OrderedDict({
                LogKeys.event: event.name,
                LogKeys.other_message: message
            })
        elif message is not None:
            message = OrderedDict({
                LogKeys.event: event.name,
                LogKeys.value: message
            })
        else:
            message = OrderedDict({
                LogKeys.event: event.name,
            })

        for k, v in kwargs.items():
            if k in LogKeys.__dict__:
                message[k] = v
        called_file, lineno = call_info
        metadata = {
            LogKeys.called_log_file: called_file,
            LogKeys.called_log_file_lineno: lineno,
            LogKeys.time_ms: self._current_time_ms,
            LogKeys.rank: self.rank
        }
        message[LogKeys.metadata] = metadata
        message = json.dumps(message)

        return self._log_template(message)

    def _log_template(self, message: str):
        return LogMeta.log_template.format(header=LogMeta.log_header,
                                           message=message)

    def get_caller(self, stacklevel=1) -> Tuple[str, int]:
        f = currentframe()

        if stacklevel == 0:
            default_file_name = f.f_code.co_filename
            default_lineno = f.f_lineno
            return (default_file_name, default_lineno)

        # On some versions of IronPython, currentframe() returns None if
        # IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv = ("(unknown file)", -1)

        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno)
            break
        return rv

    @classmethod
    def get_default_logger(cls,
                           rank: int,
                           level: LogLevel = LogLevel.SUBMITTION,
                           logger: logging.Logger = None):
        if cls._singleton is None:
            cls._singleton = cls(rank=rank, level=level, logger=logger)

        return cls._singleton
