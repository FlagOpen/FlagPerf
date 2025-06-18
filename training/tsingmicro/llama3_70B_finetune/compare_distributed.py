import sys
from tsprobe.pytorch import *

def main():
    npu_dump_path = sys.argv[1]
    gpu_dump_path = sys.argv[2]
    output_path = sys.argv[3]
    print('npu_dump_path:', npu_dump_path)
    print('gpu_dump_path:', gpu_dump_path)
    print('dump_compare_output_path:', output_path)
    compare_distributed(npu_dump_path, gpu_dump_path, output_path)

if __name__ == "__main__":
    main()