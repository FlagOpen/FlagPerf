from enum import IntEnum
from typing import Callable, NamedTuple
import inspect


class Event(IntEnum):
    INIT_START = 0
    INIT_END = 1
    TRAIN_START = 3
    TRAIN_END = 4
    EPOCH_BEGIN = 5
    EPOCH_END = 6
    STEP_BEGIN = 7
    STEP_END = 8
    BACKWARD = 9
    INIT_EVALUATION = 10
    EVALUATE = 11
    LAUNCH_TRAINING = 12
    CONVERT_MODEL = 13
    CREATE_OPTIMIZER = 14
    MODEL_TO_FP16 = 15
    MODEL_TO_DDP = 16
    FINISHED = 17
    SUBMIT_INFO = 18

    @staticmethod
    def from_string(event: str):
        try:
            return Event.__members__[event.upper()]
        except:
            raise ValueError(f"Event {event.upper()} is not available.")
        pass


class EventHandleRecord(NamedTuple):
    event: Event
    handle: Callable


class EventManager(object):
    __slot__ = ('event_handlers')

    def __init__(self) -> None:
        self.event_handlers = dict()

    def register_event_handlers(self, driver):
        for _, meth in inspect.getmembers(self, inspect.ismethod):
            prefix, _, suffix = meth.__name__.partition('on_')
            if not prefix and suffix:
                event = Event.from_string(suffix)
                ehr = EventHandleRecord(event, meth)
                driver.register_event_handler(ehr)

    def display_event_handlers(self):
        pass
