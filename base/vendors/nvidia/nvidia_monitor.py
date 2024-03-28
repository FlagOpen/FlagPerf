# ！/usr/bin/env python3
# encoding: utf-8
'''
Usage:  python3 sys-monitor.py -o operation -l [log_path]
            -o, --operation     start|stop|restart|status
            -l, --log           log path , ./logs/ default
'''

import os
import sys
import time
import signal
import atexit
import argparse
import datetime
from multiprocessing import Process
import subprocess
import schedule


class Daemon:
    '''
    daemon subprocess class.
    usage: subclass this daemon and override the run() method.
    sys-monitor.pid: in the /tmp/, auto del when unexpected exit.
    verbose: debug mode, disabled default.
    '''

    def __init__(self,
                 pid_file,
                 log_file,
                 err_file,
                 gpu_log,
                 log_path,
                 rate=5,
                 stdin=os.devnull,
                 stdout=os.devnull,
                 stderr=os.devnull,
                 home_dir='.',
                 umask=0o22,
                 verbose=0):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.home_dir = home_dir
        self.verbose = verbose
        self.pidfile = pid_file
        self.logfile = log_file
        self.errfile = err_file
        self.gpufile = gpu_log
        self.logpath = log_path
        self.rate = rate
        self.umask = umask
        self.verbose = verbose
        self.daemon_alive = True

    def get_pid(self):
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None
        except SystemExit:
            pid = None
        return pid

    def del_pid(self):
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)

    def run(self):
        '''
        NOTE: override the method in subclass
        '''

        def gpu_mon(file):
            TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            cmd = "nvidia-smi |grep 'Default'|awk '{print $3,$5,$9,$11,$13}'"
            process = subprocess.Popen(cmd,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       encoding='utf-8')
            try:
                out = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                out = process.communicate()

            if process.returncode != 0:
                result = "error"
            result = TIMESTAMP + "\n" + out[0] + "\n"
            with open(file, 'a') as f:
                f.write(result)

        def timer_gpu_mon():
            gpu_process = Process(target=gpu_mon, args=(self.gpufile, ))
            gpu_process.start()

        schedule.every(self.rate).seconds.do(timer_gpu_mon)
        while True:
            schedule.run_pending()
            time.sleep(5)

    def daemonize(self):
        if self.verbose >= 1:
            print('daemon process starting ...')
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #1 failed: %d (%s)\n' %
                             (e.errno, e.strerror))
            sys.exit(1)
        os.chdir(self.home_dir)
        os.setsid()
        os.umask(self.umask)
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #2 failed: %d (%s)\n' %
                             (e.errno, e.strerror))
            sys.exit(1)
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(self.stdin, 'r')
        so = open(self.stdout, 'a+')
        if self.stderr:
            se = open(self.stderr, 'a+')
        else:
            se = so
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        atexit.register(self.del_pid)
        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as f:
            f.write('%s\n' % pid)

    def start(self):
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)
        elif os.path.exists(self.gpufile):
            os.remove(self.gpufile)
        if self.verbose >= 1:
            print('ready to start ......')
        # check for a pid file to see if the daemon already runs
        pid = self.get_pid()
        if pid:
            msg = 'pid file %s already exists, is it already running?\n'
            sys.stderr.write(msg % self.pidfile)
            sys.exit(1)
        # start the daemon
        self.daemonize()
        self.run()

    def stop(self):
        if self.verbose >= 1:
            print('stopping ...')
        pid = self.get_pid()
        if not pid:
            msg = 'pid file [%s] does not exist. Not running?\n' % self.pidfile
            sys.stderr.write(msg)
            if os.path.exists(self.pidfile):
                os.remove(self.pidfile)
            return
        # try to kill the daemon process
        try:
            i = 0
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                i = i + 1
                if i % 10 == 0:
                    os.kill(pid, signal.SIGHUP)
        except OSError as err:
            err = str(err)
            if err.find('No such process') > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print(str(err))
                sys.exit(1)
            if self.verbose >= 1:
                print('Stopped!')

    def restart(self):
        self.stop()
        self.start()

    def status(self):
        pid = self.get_pid()
        if pid:
            if os.path.exists('/proc/%d' % pid):
                return pid
        return False


def parse_args():
    ''' Check script input parameter. '''
    parse = argparse.ArgumentParser(description='Sys monitor script')
    parse.add_argument('-o',
                       type=str,
                       metavar='[operation]',
                       required=True,
                       help='start|stop|restart|status')
    parse.add_argument('-l',
                       type=str,
                       metavar='[log_path]',
                       required=False,
                       default='./logs/',
                       help='log path')
    args = parse.parse_args()
    return args


def main():
    sample_rate1 = 5
    args = parse_args()
    operation = args.o
    log_path = args.l
    pid_fn = str('/tmp/gpu_monitor.pid')
    log_fn = str(log_path + '/nvidia_monitor.log')
    err_fn = str(log_path + '/nvidia_monitor.err')
    # result for gpu
    gpu_fn = str(log_path + '/nvidia_monitor.log')

    subdaemon = Daemon(pid_fn,
                       log_fn,
                       err_fn,
                       gpu_fn,
                       log_path,
                       verbose=1,
                       rate=sample_rate1)
    if operation == 'start':
        subdaemon.start()
    elif operation == 'stop':
        subdaemon.stop()
    elif operation == 'restart':
        subdaemon.restart()
    elif operation == 'status':
        pid = subdaemon.status()
        if pid:
            print('process [%s] is running ......' % pid)
        else:
            print('daemon process [%s] stopped' % pid)
    else:
        print("invalid argument!")
        sys.exit(1)


if __name__ == '__main__':
    main()
