import os
import sys
import time
import logging
import json
from logging import currentframe
from typing import NamedTuple, Union, Tuple, Optional
from collections import OrderedDict

from enum import IntEnum


_srcfile = os.path.normcase(logging.addLevelName.__code__.co_filename)


class LogKeys:
    default_logger_name = "PerfLogger"

    # Log format
    log_header = "PerfLog"
    log_template = "[{header}] {message}"

    # Submitted info
    submmiter: str = "submmiter"
    model: str = "model"
    optimizer_type: str = "optimizer_type"
    config: str = "config"
    config_path: str = "config_path"

    # Event
    event: str = "event"
    value: str = "value"

    # Metadata
    metadata: str = "metadata"
    called_log_file = "file"
    called_log_file_lineno = "lineno"
    time_ms = "time_ms"
    rank = "rank"

    # Other message
    other_message: str = "other"


class PerfLogLevel(IntEnum):

    INFO = 100
    SUBMITTION = 101

    @staticmethod
    def from_string(level: str):
        return PerfLogLevel.__dict__[level.upper()]

    @classmethod
    def register_to_logging(cls, logging):
        for level_name, level in PerfLogLevel.__dict__.items():
            if isinstance(level, cls):
                logging.addLevelName(level.value, level_name)


PerfLogLevel.register_to_logging(logging)


class LogEventField(NamedTuple):

    name: str
    rank: Union[int, list] = -1
    level: PerfLogLevel = PerfLogLevel.SUBMITTION


class LogEvent:

    submitted_info    = LogEventField("SUBMITTED_INFO", rank=0)
    launch_training   = LogEventField("LAUNCH_TRAINING")
    convert_model     = LogEventField("CONVERT_MODEL", rank=0)
    create_optimizer  = LogEventField("CREATE_OPTIMIZER", rank=0)
    model_to_fp16     = LogEventField("MODEL_TO_FP16", rank=0)
    model_to_ddp      = LogEventField("MODEL_TO_DDP", rank=0)
    init_start        = LogEventField("INIT_START", rank=0)
    init_end          = LogEventField("INIT_END", rank=0)
    train_begin       = LogEventField("TRAIN_BEGIN", rank=0)
    train_end         = LogEventField("TRAIN_END", rank=0)
    epoch_begin       = LogEventField("EPOCH_BEGIN", rank=0, level=PerfLogLevel.INFO)
    epoch_end         = LogEventField("EPOCH_END", rank=0, level=PerfLogLevel.INFO)
    step_begin        = LogEventField("STEP_BEGIN", rank=0, level=PerfLogLevel.INFO)
    step_end          = LogEventField("STEP_END", rank=0, level=PerfLogLevel.INFO)
    init_evaluation   = LogEventField("INIT_EVALUATION", rank=0)
    evaluation        = LogEventField("EVALUATION", rank=0)
    finished          = LogEventField("FINISHED", rank=0)

    @staticmethod
    def from_string(key: str):
        return LogEvent.__dict__[key.lower()]


class PerfLogger:

    _singleton = None

    def __init__(self, rank: int,
                 level: Union[str, PerfLogLevel]=PerfLogLevel.SUBMITTION,
                 logger: logging.Logger=None):
        self.rank = rank

        if isinstance(level, str):
            level = PerfLogLevel.from_string(level)
        self.level = level

        if logger is None:
            logger = logging.Logger(LogKeys.default_logger_name)

        self.logger = logger

        self.previous_log_time = None

    @property
    def _current_time_ms(self):
        current = int(time.time() * 1e3)
        self.previous_log_time = current
        return current

    def init_logger(self, submitter: str, model: str, config_path: str, config: dict, *args, **kwargs):
        message = {
            LogKeys.submmiter: submitter,
            LogKeys.model: model,
            LogKeys.config_path: config_path,
            LogKeys.config: config
        }

        self.log(LogEvent.submitted_info, message, *args, **kwargs)


    def log(self, event: Union[str, LogEventField], message: Optional[Union[str, dict]]=None, *args, **kwargs):
        if isinstance(event, str):
            event = LogEvent.from_string(event)

        show_log = any([
            event.rank == 0 and self.rank == 0,
            event.rank == -1,
        ]) and any([
            event.level == PerfLogLevel.SUBMITTION,
            event.level == self.level
        ])

        if not show_log:
            return

        stacklevel = 1
        if "stacklevel" in kwargs:
            stacklevel = kwargs.pop("stacklevel")

        call_info = self.get_caller(stacklevel=stacklevel)

        message = self._encode_message(event, message, call_info)
        self.logger.log(self.level.value, message, *args, **kwargs)

    def _encode_message(self, event: LogEventField,
                        message: Union[str, dict],
                        call_info: Tuple[str, int]) -> str:
        if isinstance(message, str):
            message ={LogKeys.other_message: message}
        message = OrderedDict({
            LogKeys.event: event.name,
            LogKeys.value: message
        })
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
        return LogKeys.log_template.format(header=LogKeys.log_header, message=message)

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
    def get_default_logger(cls, rank: int=-1,
                 level: Union[str, PerfLogLevel]=PerfLogLevel.SUBMITTION,
                 logger: logging.Logger=None):
        if cls._singleton is None:
            cls._singleton = cls(rank=rank, level=level, logger=logger)

        return cls._singleton















