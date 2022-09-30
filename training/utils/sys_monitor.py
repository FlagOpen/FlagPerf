# ！/usr/bin/env python3
# encoding: utf-8
'''
Usage:  python3 sys-monitor.py -o operation -l [log_path]
            -o, --operation     start|stop|restart|status
            -l, --log           log path , ./logs/ default
            -v, --gpu vendor    nvidia|iluvatar|cambricon|kunlun
'''

import os
import sys
import time
import signal
import atexit
import argparse
import schedule
import datetime
from multiprocessing import Process
from run_cmd import run_cmd_wait as rcw


class Daemon:
    '''
    daemon subprocess class.
    usage: subclass this daemon and override the run() method.
    sys-monitor.pid: in the /tmp/, auto del when unexpected exit.
    verbose: debug mode, disabled default.
    '''

    def __init__(self, pid_file, log_file, err_file, log_path, rate1=5,
                 rate2=120, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull, home_dir='.',
                 umask=0o22, verbose=0):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.home_dir = home_dir
        self.verbose = verbose
        self.pidfile = pid_file
        self.loggile = log_file
        self.errfile = err_file
        # result for cpu,mem,gpu,pwr of system
        self.log_path = log_path
        self.cpulog = str(log_path + '/cpu_monitor.log')
        self.memlog = str(log_path + '/mem_monitor.log')
        self.pwrlog = str(log_path + '/pwr_monitor.log')
        self.rate1 = rate1
        self.rate2 = rate2
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

        def cpu_mon(file):
            TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            cmd = "mpstat -P ALL 1 1|grep -v Average|grep all|awk '{print (100-$NF)/100}'"
            res, out = rcw(cmd, 10, retouts=True)
            if res:
                result = "error"
            result = TIMESTAMP + "\t" + out[0]
            with open(file, 'a') as f:
                f.write(result)

        def mem_mon(file):
            TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            cmd = "free -g|grep -i mem|awk '{print $3/$2}'"
            res, out = rcw(cmd, 10, retouts=True)
            if res:
                result = "error"
            result = TIMESTAMP + "\t" + out[0]
            with open(file, 'a') as f:
                f.write(result)

        def pwr_mon(file):
            TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            cmd = "ipmitool sdr list|grep -i Watts|awk 'BEGIN{FS = \"|\"}{for (f=1; f <= NF; f+=1) {if ($f ~ /Watts/)" \
                  " {print $f}}}'|awk '{print $1}'|sort -n -r|head -n1"
            res, out = rcw(cmd, 10, retouts=True)
            if res:
                result = "error"
            result = TIMESTAMP + "\t" + out[0]
            with open(file, 'a') as f:
                f.write(result)

        def timer_cpu_mon():
            cpu_process = Process(target=cpu_mon, args=(self.cpulog,))
            cpu_process.start()

        def timer_mem_mon():
            mem_process = Process(target=mem_mon, args=(self.memlog,))
            mem_process.start()

        def timer_pwr_mon():
            pwr_process = Process(target=pwr_mon, args=(self.pwrlog,))
            pwr_process.start()

        schedule.every(self.rate1).seconds.do(timer_cpu_mon)
        schedule.every(self.rate1).seconds.do(timer_mem_mon)
        schedule.every(self.rate2).seconds.do(timer_pwr_mon)
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
            sys.stderr.write('fork #1 failed: %d (%s)\n' % (e.errno, e.strerror))
            sys.exit(1)
        os.chdir(self.home_dir)
        os.setsid()
        os.umask(self.umask)
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #2 failed: %d (%s)\n' % (e.errno, e.strerror))
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
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        else:
            for i in self.cpulog, self.memlog, self.pwrlog:
                if os.path.exists(i):
                    os.remove(i)
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
            else:
                return False
        else:
            return False


def parse_args():
    ''' Check script input parameter. '''
    parse = argparse.ArgumentParser(description='Sys monitor script')
    parse.add_argument('-o', type=str, metavar='[operation]', required=True,
                       help='start|stop|restart|status')
    parse.add_argument('-l', type=str, metavar='[log_path]', required=False,
                       default='./logs/', help='log path')
    args = parse.parse_args()
    return args


def main():
    sample_rate1 = 5
    sample_rate2 = 120
    args = parse_args()
    operation = args.o
    path = args.l
    pid_fn = str('/tmp/sys_monitor.pid')
    log_fn = str(path + '/sys_monitor.log')
    err_fn = str(path + '/sys_monitor.err')

    subdaemon = Daemon(pid_fn, log_fn, err_fn, path, verbose=1, rate1=sample_rate1,
                       rate2=sample_rate2)
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
