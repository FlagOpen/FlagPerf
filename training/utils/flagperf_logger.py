# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
'''Log module for flagperf
   There are 4 log levels:
   - DEBUG: Very detail messages for debugging.
   - INFO: Normal infos.
   - WARNING: Something is wrong but critical, we can continue.
   - ERROR: Something is wrong, we have to stop.
'''

import os
import logging
import datetime
import sys

LOGLEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}


class ColorFormatter():
    '''A color formatter for logging module.'''
    COLORS_TABLE = {
        'grey': "\033[0;37;40m",
        'white': "\033[1;37;40m",
        'yellow': "\033[1;33;40m]",
        'red': "\033[1;31;40m",
        'color_end': "\033[0m"
    }

    LOG_COLORS = {
        'DEBUG': COLORS_TABLE['grey'],
        'INFO': COLORS_TABLE['white'],
        'WARNING': COLORS_TABLE['yellow'],
        'ERROR': COLORS_TABLE['red']
    }

    def __init__(self, log_caller):
        if log_caller:
            self.log_format = "%(asctime)s\t[%(levelname)s]\t[%(meta)s]%(message)s"
        else:
            self.log_format = "%(asctime)s\t[%(levelname)s]\t%(message)s"

    def format(self, record):
        '''Return formatter for color log handler.'''
        color_log_format = self.LOG_COLORS[record.levelname] \
                           + self.log_format + self.COLORS_TABLE['color_end']
        formatter = logging.Formatter(color_log_format)
        return formatter.format(record)


def _create_log_file(log_dir, log_file):
    '''Create logdir and logfile if they don't exist.'''
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    curr_log_file = os.path.join(log_dir, log_file)

    file_d = open(curr_log_file, mode='a+', encoding='utf-8')
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_d.write(time_str + " FlagperfLogger started.\n")
    file_d.close()
    return curr_log_file


def _get_caller():
    curr_frame = logging.currentframe()
    # if curr_frame is not None:
    #    curr_frame = curr_frame.f_back
    caller = ("(unknown file)", 0)
    while hasattr(curr_frame, "f_code"):
        code_obj = curr_frame.f_code
        filename = os.path.normcase(code_obj.co_filename)
        if filename in (__file__, logging._srcfile):
            curr_frame = curr_frame.f_back
            continue
        filename = os.path.basename(filename)
        caller = (filename, curr_frame.f_lineno)
        break
    return caller


def logger_print(method):

    def wrapper(self, msg):
        sys.stdout = self.stdout
        method(self, msg)
        sys.stdout = self.nullout

    return wrapper


class FlagPerfLogger():
    '''A logger for benchmark.'''

    def __init__(self):
        '''Arguments:
        '''
        self.perf_logger = None
        self.mode = None
        self.logfile_handler = None
        self.console_handler = None
        self.logpath = None
        self.logfile = None
        self.log_caller = False

    def init(self,
             logpath,
             logfile,
             loglevel='info',
             mode="file",
             stdout=sys.stdout,
             nullout=sys.stdout,
             log_caller=False):
        '''Set log level and create the log file.
           Arguments:
           - logpath: the path to put the logfile
           - logfile: logfile name
           - loglevel: defaut is INFO
           - mode: Default is 'file'.
                - 'file': only log to logfile.
                - 'console': only print log to console and ignore log file
                - 'both': print log to console and log to logfile
        '''
        self.perf_logger = logging.getLogger()
        self.mode = mode
        if log_caller:
            log_format = "%(asctime)s\t[%(levelname)s]\t[%(meta)s]%(message)s"
        else:
            log_format = "%(asctime)s\t[%(levelname)s]\t%(message)s"
        handler_formatter = logging.Formatter(log_format)
        color_handler_formatter = ColorFormatter(log_caller)
        log_level = LOGLEVELS.get(loglevel, logging.INFO)
        self.perf_logger.setLevel(log_level)
        self.log_caller = log_caller
        if self.mode == "file" or self.mode == "both":
            self.logpath = logpath
            curr_log_file = _create_log_file(logpath, logfile)
            self.logfile_handler = logging.FileHandler(curr_log_file,
                                                       'a',
                                                       encoding='utf-8')
            self.logfile_handler.setFormatter(handler_formatter)
            self.perf_logger.addHandler(self.logfile_handler)

        if self.mode == "console" or self.mode == "both":
            self.console_handler = logging.StreamHandler()
            self.console_handler.setFormatter(color_handler_formatter)
            self.perf_logger.addHandler(self.console_handler)
        sys.stdout = nullout
        self.stdout = stdout
        self.nullout = nullout

    def stop(self):
        '''If any file opened, close here. Then remove handdlers.'''
        self.info("Stop FlagperfLogger.")
        if self.mode == "both" or self.mode == "file":
            self.logfile_handler.close()
            self.perf_logger.removeHandler(self.logfile_handler)
        if self.mode == "both" or self.mode == "console":
            self.perf_logger.removeHandler(self.console_handler)

    @logger_print
    def info(self, msg):
        '''Call logging.info() to log time, log level and msg.'''
        if self.log_caller:
            file_name, line_no = _get_caller()
            caller_info = file_name + "," + str(line_no)
            self.perf_logger.info(msg, extra={'meta': caller_info})
        else:
            self.perf_logger.info(msg)

    @logger_print
    def warning(self, msg):
        '''Call logging.warning() to log time, log level and msg.'''
        if self.log_caller:
            file_name, line_no = _get_caller()
            caller_info = file_name + "," + str(line_no)
            self.perf_logger.warning(msg, extra={'meta': caller_info})
        else:
            self.perf_logger.warning(msg)

    @logger_print
    def debug(self, msg):
        '''Call logging.debug() to log time, log level and msg.'''
        if self.log_caller:
            file_name, line_no = _get_caller()
            caller_info = file_name + "," + str(line_no)
            self.perf_logger.debug(msg, extra={'meta': caller_info})
        else:
            self.perf_logger.debug(msg)

    @logger_print
    def error(self, msg):
        '''Call logging.error() to log time, log level and msg.'''
        if self.log_caller:
            file_name, line_no = _get_caller()
            caller_info = file_name + "," + str(line_no)
            self.perf_logger.error(msg, extra={'meta': caller_info})
        else:
            self.perf_logger.error(msg)
