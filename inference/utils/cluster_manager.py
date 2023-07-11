# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
'''Cluster Manager'''

import os
import sys

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURR_PATH))
import run_cmd
from loguru import logger


class ClusterManager():
    '''A cluster manager that can make healthcheck, distribute files, and run a
       command in the cluster.
    '''

    def __init__(self):
        self.hosts = None
        self.ssh_port = None
        self.user = None
        self.ssh_cmd_head = None
        self.scp_cmd_head = None

    def init(self, hosts, port, user):
        '''Init with all args that ssh needs.'''
        self.hosts = hosts
        self.ssh_port = port
        self.user = user
        self.ssh_cmd_head = "ssh -o ConnectTimeout=3" \
                            + " -o StrictHostKeyChecking=no -l " + self.user \
                            + " -p " + port
        self.scp_cmd_head = "scp -o  ConnectTimeout=3 " \
                            + "-o StrictHostKeyChecking=no -P " + port

        logger.debug(f"ssh: {self.ssh_cmd_head}")
        logger.debug(f"scp: {self.scp_cmd_head}")

    def _run_command_ssh_remote(self, cmd, host, timeout=10):
        ''' Run cmd on host with ssh.
            Return exit code of cmd and stdout/stderr messages.
        '''
        ssh_run_cmd = self.ssh_cmd_head + " " + host + " \'" + cmd + "\'"
        logger.debug("Run cmd on host with ssh. ssh cmd=" + cmd + " host=" +
                     host + " timeout=" + str(timeout))
        ret, outs = run_cmd.run_cmd_wait(ssh_run_cmd, timeout)
        return ret, outs

    def healthcheck(self):
        '''Return the hosts not alive.
        '''
        return self.run_command_all_hosts(":")

    def get_hosts_list(self):
        '''Return the lists of all hosts.'''
        return self.hosts

    def get_hosts_count(self):
        '''Return count of the hosts.
        '''
        return len(self.hosts)

    def run_command_all_hosts(self, command, timeout=10):
        '''Run a command on each host with ssh.
        '''
        failed_hosts_ret = {}
        for host in self.hosts:
            ret, outs = self._run_command_ssh_remote(command, host, timeout)
            if ret != 0:
                failed_hosts_ret[host] = ret
                logger.error("Run cmd on host " + host + " cmd=" + command +
                             " [FAILED]. Output: " + outs[0])
        return failed_hosts_ret

    def run_command_some_hosts(self,
                               command,
                               host_count=1,
                               timeout=10,
                               no_log=False):
        '''Run a command on each host with ssh.
        '''
        failed_hosts_ret = {}
        for i in range(0, host_count):
            logger.debug("host number:" + str(i))
            host = self.hosts[i]
            ret, outs = self._run_command_ssh_remote(command, host, timeout)
            if ret != 0:
                failed_hosts_ret[host] = ret
                if not no_log:
                    logger.error("Run cmd on host " + host + " cmd=" +
                                 command + " [FAILED]. Output: " + outs[0])
        return failed_hosts_ret

    def start_monitors_some_hosts(self,
                                  base_command,
                                  case_log_dir,
                                  host_count,
                                  timeout=10):
        '''Start monitors on hosts with ssh.
        '''
        failed_hosts_ret = {}
        for i in range(0, host_count):
            logger.debug("host number:" + str(i))
            host = self.hosts[i]
            # add log_dir option to the command
            log_dir = os.path.join(case_log_dir, host + "_noderank" + str(i))
            command = base_command + log_dir
            ret, outs = self._run_command_ssh_remote(command, host, timeout)
            if ret != 0:
                failed_hosts_ret[host] = ret
                logger.error("Run cmd on host " + host + " cmd=" + command +
                             " [FAILED]. Output: " + outs[0])
        return failed_hosts_ret

    def run_command_some_hosts_distribution_info(self,
                                                 base_cmd,
                                                 host_count,
                                                 timeout=10):
        '''Run a command with torch ddp options on each host with ssh.
        '''
        failed_hosts_ret = {}
        # remove the " at the end of base_cmd, then add other options.
        # base_cmd = base_cmd.rstrip(" \"")
        # command_master_ip = base_cmd + ' --master_addr ' + self.hosts[0]
        for i in range(0, host_count):
            host = self.hosts[i]
            ret, outs = self._run_command_ssh_remote(base_cmd, host, timeout)

            if ret != 0:
                failed_hosts_ret[host] = ret
                logger.debug("Run cmd on host " + host + " cmd=" + base_cmd +
                             " node_rank=" + str(i) + " [FAILED]. Output: " +
                             outs[0])
        return failed_hosts_ret

    def _scp_file_to_remote_host(self,
                                 host,
                                 local_file,
                                 remote_dir,
                                 timeout=600):
        ''' Run scp command to copy local_file to remote_dir.
        '''

        scp_cmd = self.scp_cmd_head + " " + local_file + " " + self.user \
                                    + "@" + host + ":" + remote_dir + "/"
        logger.debug("scp command:" + scp_cmd)
        ret, outs = run_cmd.run_cmd_wait(scp_cmd, timeout)
        return ret, outs

    def sync_file_to_some_hosts(self,
                                local_file,
                                remote_dir,
                                host_count,
                                timeout=600):
        '''scp local_file to remote_dir on hosts in the cluster .'''
        failed_hosts_ret = {}
        if not os.path.exists(local_file):
            logger.error("Can't find local file before scp:" + local_file)
            for host in self.hosts:
                failed_hosts_ret[host] = 1
            return failed_hosts_ret

        for i in range(0, host_count):
            host = self.hosts[i]
            ret, outs = self._scp_file_to_remote_host(host,
                                                      local_file,
                                                      remote_dir,
                                                      timeout=timeout)
            if ret != 0:
                failed_hosts_ret[host] = ret
                logger.debug("Scp local file " + local_file + "to " + host +
                             ":" + remote_dir + " [FAILED]. Output: " +
                             outs[0])
        return failed_hosts_ret

    def _scp_dir_from_remote_host(self,
                                  host,
                                  remote_dir,
                                  local_dir,
                                  timeout=600):
        ''' Run scp command to copy remote_dir to local_dir.
        '''
        scp_cmd = self.scp_cmd_head + " -r " + self.user + "@" + host + ":" \
                                    + remote_dir + "/* " + local_dir + "/"
        logger.debug("scp command:" + scp_cmd)
        ret, outs = run_cmd.run_cmd_wait(scp_cmd, timeout)
        return ret, outs

    def collect_files_some_hosts(self,
                                 remote_dir,
                                 local_dir,
                                 host_count,
                                 timeout=600):
        '''scp remote_dir from hosts in the cluster to <local_dir>/<host>.
        '''
        failed_hosts_ret = {}
        for i in range(0, host_count):
            host = self.hosts[i]
            if not os.path.exists(local_dir):
                logger.debug("Make local dir:" + local_dir)
                os.makedirs(local_dir)
            ret, outs = self._scp_dir_from_remote_host(host,
                                                       remote_dir,
                                                       local_dir,
                                                       timeout=timeout)
            if ret != 0:
                failed_hosts_ret[host] = ret
                logger.debug("Scp from " + host + ":" + remote_dir + " to " +
                             local_dir + " [FAILED]. Output: " + outs[0])
        return failed_hosts_ret
