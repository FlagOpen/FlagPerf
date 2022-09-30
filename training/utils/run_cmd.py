# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
'''Basic functions to run shell commands'''

import subprocess


def run_cmd_wait(cmd, timeout, retouts=True):
    '''Run a shell command and wait <timeout> second(s).'''
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='utf-8')
    try:
        output = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output = process.communicate()

    if retouts:
        return [process.returncode, output]
    return process.returncode
