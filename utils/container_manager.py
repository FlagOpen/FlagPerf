# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
'''Container Manager'''

import os
import sys
from argparse import ArgumentParser

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURR_PATH))
import run_cmd


class ContainerManager():
    '''A container manager that can start/stop/remove a container, run a
       command in container, check container status with docker command.
    '''

    def __init__(self, container_name):
        self.name = container_name

    def run_new(self, container_run_args, docker_image):
        '''Start a new docker container with <container_run_args>'''
        exists = self.exists()
        if exists is True:
            return 1, ["Conatiner exists.", None]

        run_new_cmd = "docker run " + container_run_args + \
                      " --name=" + self.name + " \"" + docker_image + "\" " + \
                      "sleep infinity"
        print(run_new_cmd)
        ret, outs = run_cmd.run_cmd_wait(run_new_cmd, 10)
        return ret, outs

    def run_cmd_in(self, cmd_in_container, timeout=5, detach=True):
        '''Start a new docker container with <container_run_args>'''
        if os.getenv("EXEC_IN_CONTAINER", False):
            ret, outs= run_cmd.run_cmd_wait(cmd_in_container, 15)
            return ret, outs
        exists = self.exists()
        if exists is False:
            return 1, ["Conatiner doesn't exist.", None]

        if detach:
            exec_cmd_head = "docker exec -d "
        else:
            exec_cmd_head = "docker exec -i "

        exec_cmd = exec_cmd_head + self.name + " bash -c \"" \
                                             + cmd_in_container + "\""
        print("run cmd in:", exec_cmd)
        ret, outs = run_cmd.run_cmd_wait(exec_cmd, timeout)
        print("ret:", ret, " outs:", outs[0])
        return ret, outs

    def start(self):
        '''Start the stopped container. Useless now.'''
        exists = self.exists()
        if exists is False:
            return 1, ["Conatiner doesn't exist.", None]

        rm_cmd = "docker start " + self.name
        ret, outs = run_cmd.run_cmd_wait(rm_cmd, 3)
        return ret, outs

    def stop(self):
        '''Stop the container.'''
        exists = self.exists()
        if exists is False:
            return 0, ["Conatiner doesn't exist.", None]

        rm_cmd = "docker stop " + self.name
        ret, outs = run_cmd.run_cmd_wait(rm_cmd, 10)
        return ret, outs

    def remove(self):
        '''Remove the container. Useless now.'''
        exists = self.exists()
        if exists is False:
            return 0, ["", None]

        rm_cmd = "docker rm -f " + self.name
        ret, outs = run_cmd.run_cmd_wait(rm_cmd, 3)
        return ret, outs

    def exists(self):
        '''Return whether the container exists.
           Return value:
               True: It exists.
               False: It doesn't exist.
        '''
        exists = None
        check_cmd = "docker ps -a | grep " + self.name + "$ | wc -l"
        ret, outs = run_cmd.run_cmd_wait(check_cmd, 3)

        if ret == 0:
            if str(outs[0]) == "1\n":
                exists = True
            elif str(outs[0]) == "0\n":
                exists = False
        return exists

    def is_pid_running(self, pid_file_path):
        '''Return whether the process with pid is running in container.
           Return value:
               True: It is running.
               False: It isn't running.
        '''
        get_pid_cmd = "cat " + pid_file_path
        ret, outs = self.run_cmd_in(get_pid_cmd, detach=False)
        if ret == 0:
            task_pid = int(outs[0])
        else:
            print("Can't find pid file ", pid_file_path, "in container.")
            return False
        check_cmd = "ls /proc/" + str(task_pid) + "/cmdline"
        ret, outs = self.run_cmd_in(check_cmd, detach=False)
        if ret == 0:
            print("The process is running.")
            return True
        print("The process is not running.")
        return False


def _parse_args():
    '''Get command args from input. '''
    parser = ArgumentParser(description="Manage a container. ")
    parser.add_argument("-o",
                        type=str,
                        required=True,
                        choices=[
                            'start', 'stop', 'rm', 'exists', 'runnew',
                            'runcmdin', 'pidrunning'
                        ],
                        help="Operation on the container:"
                        "start    Start a stopped container."
                        "stop     Stop a container."
                        "rm       Remove a container."
                        "exists   Check whether a container exists."
                        "runnew   Start a new container with run args."
                        "runcmdin Run a command in the container."
                        "pidrunning Check wether the process is running.")
    parser.add_argument("-c", type=str, required=True, help="Container name")

    args, _ = parser.parse_known_args()

    if args.o == 'runnew':
        parser.add_argument("-i",
                            type=str,
                            required=True,
                            help="Docker image.")
        parser.add_argument("-a",
                            type=str,
                            required=True,
                            help="container start args.")
    elif args.o == 'runcmdin':
        parser.add_argument("-r",
                            type=str,
                            required=True,
                            help="command to run")
        parser.add_argument("-d",
                            action='store_true',
                            default=False,
                            help="command to run")
        parser.add_argument("-t",
                            type=int,
                            default=60,
                            help="timeout of running")
    elif args.o == 'pidrunning':
        parser.add_argument("-f",
                            type=str,
                            required=True,
                            help="pid file path in container.")
    args = parser.parse_args()
    return args


def main():
    '''Support command line for container manager. Called by cluster manager.
    '''
    args = _parse_args()
    container_mgr = ContainerManager(args.c)
    operation = args.o
    ret = None
    outs = None

    if operation == "exists":
        if container_mgr.exists():
            print("Container exists.")
            sys.exit(0)
        print("Container doesn't exist.")
        sys.exit(1)

    if operation == "pidrunning":
        if container_mgr.is_pid_running(args.f):
            sys.exit(0)
        sys.exit(1)

    if operation == "start":
        ret, outs = container_mgr.start()
    elif operation == "stop":
        ret, outs = container_mgr.stop()
    elif operation == "rm":
        ret, outs = container_mgr.remove()
    elif operation == "runcmdin":
        cmd = args.r
        detach = args.d
        timeout = args.t
        ret, outs = container_mgr.run_cmd_in(cmd, timeout, detach)
    elif operation == "runnew":
        run_args = args.a
        docker_image = args.i
        ret, outs = container_mgr.run_new(run_args, docker_image)

    if ret == 0:
        print("Output: ", outs[0])
        print(operation, "successful.")
        sys.exit(0)
    print("Output: ", outs[0])
    print(operation, "failed.")
    sys.exit(ret)


if __name__ == '__main__':
    main()
