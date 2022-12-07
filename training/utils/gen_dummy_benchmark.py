# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
'''Generate a dummy benchmark in benchmark/dummy/<framework>. Put an empty
   config file in <vendor>/config/config_xxx.py.'''
import os
import sys
from argparse import ArgumentParser


CURR_PATH = os.path.abspath(os.path.dirname(__file__))


class DummyBenchmark():
    '''A dummy benchmark that can be added to FlagPerf.'''
    CONFIG_CONTENT = '''
dummy_config1 = True
dummy_config2 = False
dummy_config3 = 99
'''
    PRETRAIN_CONTENT = '''
import sys
print("Get options:", str(sys.argv))
current_env = os.environ.copy()
for environ in current_env.keys():
     print(environ + ":" + current_env[environ])
'''
    def __init__(self, vendor, framework, config_file, data_dir):
        self.vendor = vendor
        self.framework = framework
        self.config_file = config_file
        self.data_dir = data_dir

    def _get_vendor_config_dir(self):
        '''Return dir of config in the vendor's dir.'''
        return os.path.join(CURR_PATH, "../" + self.vendor,
                            "dummy-" + self.framework, "config")

    def _get_benchmark_dir(self):
        '''Return dir of the benchmark code.'''
        return os.path.join(CURR_PATH, "../benchmarks/dummy/", self.framework)

    def _write_file(self, file_path, content):
        '''Write the content to file_path.'''
        file_o = open(file_path, "w")
        file_o.write(content)
        file_o.close()

    def _test_and_makedirs(self, dest_dir):
        '''Make dirs if <dest_dir> doesn't exist.'''
        if os.path.isdir(dest_dir):
            return True
        if os.path.isfile(dest_dir):
            print("Error! ", dest_dir, " is a file!")
            return False
        os.makedirs(dest_dir)

    def add_to_perf(self):
        ''''Add dummy benchmark code to FlagPerf.'''
        benchmark_code_dir = self._get_benchmark_dir()
        vendor_config_dir = self._get_vendor_config_dir()
        self._test_and_makedirs(benchmark_code_dir)
        self._test_and_makedirs(vendor_config_dir)
        self._test_and_makedirs(self.data_dir)
        vendor_config_file = os.path.join(vendor_config_dir,
                                          self.config_file + ".py")
        self._write_file(vendor_config_file, self.CONFIG_CONTENT)
        pretrain_file = os.path.join(benchmark_code_dir, "run_pretrain.py")
        self._write_file(pretrain_file, self.PRETRAIN_CONTENT)

        print("You can run this command to clear dummy benchmark:")
        print("rm -rf ", benchmark_code_dir, vendor_config_dir, self.data_dir)
        print("=======================================================")

    def print_dummy_test_conf(self):
        print("You can add the dummy benchmark case in test_conf like this:")
        print("DUMMY_TEST = {")
        print("    \"model\": \"dummy\"")
        print("    \"framework\": \"" + self.framework + "\"")
        print("    \"config\": \"" + self.config_file + "\"")
        print("    \"repeat\": <times to run>")
        print("    \"nnodes\": <hosts count>")
        print("    \"nproc\": <count of processes on each host>")
        print("    \"data_dir_host\": \"", self.data_dir, "\"")
        print("    \"data_dir_container\": \"/mnt/data/dummy\"")
        print("}")


def _parse_args():
    '''Get command args from input. '''
    parser = ArgumentParser(description="Generate a dummy benchmark case.")
    parser.add_argument("-v", type=str, metavar='[vendor]', required=True, help="Vendor name")
    parser.add_argument("-f", type=str, metavar='[framework]', required=True, help="Framework")
    parser.add_argument("-c", type=str, metavar='[config]', required=True, 
                        help="Config file name, e.g. config_A100_1x8.")
    parser.add_argument("-d", type=str, metavar='[data dir]', required=True, help="Dummy data dir")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    dummy_benchmark = DummyBenchmark(args.v, args.f, args.c, args.d)
    dummy_benchmark.add_to_perf()
    dummy_benchmark.print_dummy_test_conf()

if __name__ == '__main__':
    main()
