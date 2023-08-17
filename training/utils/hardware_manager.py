import argparse
import os
from run_cmd import run_cmd_wait


def _parse_args():
    ''' Check script input parameter. '''
    parse = argparse.ArgumentParser(description='Hardware helper script')
    parse.add_argument('-p',
                       type=str,
                       metavar='[path]',
                       required=False,
                       default=".",
                       help='output path')

    parse.add_argument('-o',
                       type=str,
                       metavar='[outfile]',
                       required=False,
                       default="hardware_info.txt",
                       help='filename of hardware info')
    args = parse.parse_args()
    return args


class HardwareManager:

    def __init__(self):
        self.case_log_dir = None

    def init(self, case_log_dir):
        self.case_log_dir = case_log_dir

    # 示例 Intel(R) Xeon(R) Gold 6346 CPU @ 3.10GHz
    @classmethod
    def get_cpu_info(cls) -> str:
        cmd = "sudo lscpu"
        retcode, output = run_cmd_wait(cmd, timeout=10)
        if retcode != 0:
            return "failed to get cpu info"
        cpu_info = ""
        for line in output[0].split("\n"):
            if "Model name" in line:
                kv = line.split(":")
                cpu_info = kv[1].strip()
        return cpu_info

    @classmethod
    def get_total_memory(cls) -> str:
        cmd = "sudo free -g"
        retcode, output = run_cmd_wait(cmd, timeout=10)
        if retcode != 0:
            return "failed to get memory info"

        mem_GiB = 0.0
        for row_idx, line in enumerate(output[0].split("\n")):
            if row_idx != 1:
                continue
            items = line.split()
            mem_GiB = round(float(items[1]) / 1024, 2)
        return f"{mem_GiB} GiB"

    # 服务器信息(厂商 + 机型) 例如：Inspur NF5468M6
    @classmethod
    def get_server_info(cls):
        cmd = "sudo dmidecode -t system"
        retcode, output = run_cmd_wait(cmd, timeout=10)
        if retcode != 0:
            return f"failed to get server info: retcode: {retcode}"

        manufacturer = ""
        product_name = ""

        for line in output[0].split("\n"):
            if "Manufacturer" in line:
                kv = line.split(":")
                manufacturer = kv[1].strip()

            if "Product Name" in line:
                kv = line.split(":")
                product_name = kv[1].strip()

        return f"{manufacturer} {product_name}"

    @classmethod
    def get_os_kernerl_info(cls):
        cmd = "sudo uname -r"
        retcode, output = run_cmd_wait(cmd, timeout=10)
        if retcode != 0:
            return f"failed to get kernel info: retcode: {retcode}"
        os_kernel = output[0].strip()
        return f"linux {os_kernel}"

    @classmethod
    def get_docker_version(cls):
        cmd = "sudo docker --version"
        retcode, output = run_cmd_wait(cmd, timeout=10)
        if retcode != 0:
            return f"failed to get docker version: retcode: {retcode}"

        docker_version = output[0].split(",")[0].strip()
        return docker_version

    def write_hardware_info(self, out_dir):
        hardware_info_list = list()
        hardware_info_list.append("server: " + self.get_server_info())
        hardware_info_list.append("cpu: " + self.get_cpu_info())
        hardware_info_list.append("memory: " + self.get_total_memory())
        hardware_info_list.append("os_kernel: " + self.get_os_kernerl_info())
        hardware_info_list.append("docker: " + self.get_docker_version())
        hardware_info_list.append("")

        output_filename = "hardware_info.txt"

        content = "\n".join(hardware_info_list)
        output_file = os.path.join(self.case_log_dir, output_filename)
        write_file(content, output_file)


def main():
    args = _parse_args()
    path = args.p
    hm = HardwareManager()
    hm.init(path)
    hm.write_hardware_info(path)


def write_file(content, file):
    with open(file, "w") as f:
        f.write(content)
    file_size = get_filesize(file)
    if file_size > 0:
        print(f"write output file succeed. filepath: {file} size: {file_size}")


def get_filesize(file):
    if not os.path.isfile(file):
        return 0
    size = os.stat(file).st_size
    return size


def get_task_log_dir(task_args):
    task_log_dir = os.path.join(
        task_args.log_dir,
        task_args.case_name + "/" + "round" + str(task_args.round) + "/" +
        task_args.host_addr + "_noderank" + str(task_args.node_rank))
    return task_log_dir


if __name__ == "__main__":
    main()
