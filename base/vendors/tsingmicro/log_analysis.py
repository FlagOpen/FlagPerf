import os
import argparse

def parse_args():
    """
    :return: pare args
    """
    parser = argparse.ArgumentParser(description='log analysis')
    parser.add_argument("--log_type", type=str, help='the log type')
    parser.add_argument("--log_file", type=str, help='the log file path')

    args = parser.parse_args()
    return args

def ddr_perf_analysis(log_file):
    read_bw = 0
    write_bw = 0
    count = 0
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            if "ddr_bandwidth" in line:
                line = line.split(':')[-1].strip()
                line = line.split('=')
                read_bw_str  = line[1].strip()
                write_bw_str = line[-1].strip()
                read_bw  += int(read_bw_str.split('M')[0].strip())
                write_bw += int(write_bw_str.split('M')[0].strip())
                count += 1

                if count == 4:
                    bandwidth = write_bw if write_bw > read_bw else read_bw
                    print(f"[FlagPerf Result]main_memory-bandwidth={round(bandwidth/1024, 2)} GB/s")
                    break

def ddr_cap_analysis(log_file):
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            # |   0   TX81  19830   54C     95W / 200W  |  00000000:89:00.0  |    5508M /137072M  |      37944 |
            if "TX81" in line:
                line = line.split('|')[-3].strip()
                mem_used  = int(line.split('/')[0].split('M')[0].strip())
                mem_total = int(line.split('/')[1].split('M')[0].strip())
                mem_cap = (mem_total - mem_used + 1024) / 1024 * 4
                print(f"[FlagPerf Result]main_memory-capacity={int(mem_cap)} GB")
                break

def c2c_global_latency_analysis(log_file):
    with open(log_file, "r") as f_log:
        result_max = 0
        while True:
            line = f_log.readline()
            if not line: break
            # c2c latency test: [0]=0x452 [1]=0x452 [2]=0x450 [3]=0x454 [4]=0xb11 [5]=0xb15 [6]=0x456 [7]=0x454
            if "c2c latency test:" in line:
                result_list_temp = line.split('=')
                for result_temp in result_list_temp:
                    if "latency" in result_temp:
                        continue

                    result_str = result_temp.split('[')[0].strip()
                    result_num = int(result_str, 16)
                    if result_num > result_max:
                        result_max = result_num

                result_max = int(result_max/2)
                print(f"[FlagPerf Result] c2c global latency={result_max} ns")
                break

def c2c_global_perf_analysis(log_file):
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            if "c2c_test: bandwidth" in line:
                line = line.split('=')[-1].strip()
                c2c_bw = int(line.split('Mb')[0].strip())
                c2c_bw /= 8
                c2c_bw /= 1024

                print(f"[FlagPerf Result]interconnect-P2P_interserver={int(c2c_bw)} GB/s")
                break

def c2c_perf_analysis(log_file):
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            if "c2c_test: bandwidth" in line:
                line = line.split('=')[-1].strip()
                c2c_bw = int(line.split('Mb')[0].strip())
                # c2c_bw /= 8
                c2c_bw /= 1024

                print(f"[FlagPerf Result]interconnect-P2P_intraserver={int(c2c_bw * 2)} GB/s")  # send + recv
                break

def pcie_perf_analysis(log_file):
    with open(log_file, "r") as f_log:
        count = 0
        h2d_bw = 0
        d2h_bw = 0
        while True:
            line = f_log.readline()
            if not line: break

            # [2024-12-03 06:53:34]: pcie_test: bandwidth_h2d = 14 GB/s
            # [2024-12-03 06:53:34]: pcie_test: bandwidth_d2h = 14 GB/s
            if "bandwidth_h2d" in line:
                line = line.split(':')[-1].strip().split('=')[-1]
                h2d_bw += int(line.split('G')[0].strip())

            if "bandwidth_d2h" in line:
                line = line.split(':')[-1].strip().split('=')[-1]
                d2h_bw += int(line.split('G')[0].strip())
                count += 1
                if count == 4:
                    print(f"[FlagPerf Result]interconnect-h2d={h2d_bw} GB/s")
                    print(f"[FlagPerf Result]interconnect-d2h={d2h_bw} GB/s")
                    break

def computation_analysis(log_file, type_str):
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            # [gemm_dyn.cpp:GetGemmCycle:292] !!! gemm total_perf:175.252 Tops, Percentage: 98.210 %
            if "total_perf" in line:
                line = line.split(':')[-2].strip()
                computation_value = float(line.split('T')[0].strip()) * 4
                if "INT" in type_str:
                    print(f"[FlagPerf Result]{type_str}={round(computation_value, 2)}Tops")
                else:
                    print(f"[FlagPerf Result]{type_str}={round(computation_value, 2)}TFlops")
                break

def power_full_load_analysis(log_file):
    with open(log_file, "r") as f_log:
        core_power_max = 0
        while True:
            line = f_log.readline()
            if not line: break

            # Power info: PowerSum: 119W, MaxPower: 31W, MinPower: 29W, MaxTemp: 55C
            if "PowerSum:" in line:
                # core_power = float(line.split('W')[0].strip()[-4:].strip())
                core_power = float(line.split(':')[2].strip().split('W')[0])
                if (core_power < (200 * 4)):
                    core_power_max = core_power if core_power > core_power_max else core_power_max

        core_power = core_power_max / 8
        print(f"[FlagPerf Result]power-full_load(core-power)={round(core_power, 2)} W")

def allreduce_analysis(log_file, type_str):
    with open(log_file, "r") as f_log:
        while True:
            line = f_log.readline()
            if not line: break

            if "allreduce perf:" in line:
                perf_str = line.split('GB')[0].split(':')[-1].strip()
                allreduce_perf = float(perf_str)

                if "intra" in type_str:
                    print(f"[FlagPerf Result]interconnect-P2P_intraserver-bandwidth={allreduce_perf} GB/s")
                else:
                    print(f"[FlagPerf Result]interconnect-P2P_interserver-bandwidth={allreduce_perf} GB/s")
                break

if __name__ == '__main__':
    args = parse_args()

    if args.log_type == "ddr_perf":
        ddr_perf_analysis(args.log_file)
    elif args.log_type == "ddr_cap":
        ddr_cap_analysis(args.log_file)
    elif args.log_type == "c2c_perf":
        c2c_perf_analysis(args.log_file)
    elif args.log_type == "c2c_global_perf":
        c2c_global_perf_analysis(args.log_file)
    elif args.log_type == "pcie_perf":
        pcie_perf_analysis(args.log_file)
    elif args.log_type == "computation_bf16":
        computation_analysis(args.log_file, "computation-BF16")
    elif args.log_type == "computation_fp16":
        computation_analysis(args.log_file, "computation-FP16")
    elif args.log_type == "computation_int8":
        computation_analysis(args.log_file, "computation-INT8")
    elif args.log_type == "computation_tf32":
        computation_analysis(args.log_file, "computation-TF32")
    elif args.log_type == "computation_fp32":
        computation_analysis(args.log_file, "computation-FP32")
    elif args.log_type == "power_full_load":
        power_full_load_analysis(args.log_file)
    elif args.log_type == "intra_allreduce":
        allreduce_analysis(args.log_file, "intra_allreduce")
    elif args.log_type == "inter_allreduce":
        allreduce_analysis(args.log_file, "inter_allreduce")

    print()
