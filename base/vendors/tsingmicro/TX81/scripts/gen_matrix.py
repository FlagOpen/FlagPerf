import numpy as np
import argparse
import os
import sys
import logging
import time

def generate_matrix(m, n, data_format, output_file=None):
    """
    生成一个 m x n 矩阵，并以指定的数据格式保存为 .bin 文件。

    参数:
    - m (int): 矩阵的行数。
    - n (int): 矩阵的列数。
    - data_format (str): 数据格式，支持 "int8", "half", "bfloat16", "tfloat"。
    - output_file (str): 输出二进制文件的路径。如果未指定，将生成唯一的文件名。
    """
    logging.info(f"Generating a {m}x{n} matrix with format {data_format}.")

    if data_format == "int8":
        # 生成只包含 -1 和 1 的随机整数矩阵
        matrix_converted = np.random.choice([-1, 1], size=(m, n)).astype(np.int8)
    else:
        # 对于其他数据格式，生成 [0, 1) 范围内的随机浮点数矩阵
        matrix = np.random.rand(m, n).astype(np.float32)

        if data_format == "half":
            # 转换为 float16（半精度浮点数）
            matrix_converted = matrix.astype(np.float16)
        elif data_format == "bfloat16":
            # 将 float32 转换为 bfloat16
            # bfloat16 通常由 float32 的高 16 位组成
            matrix_fp32 = matrix.copy()
            # 将 float32 数据视为 uint32 以进行位操作
            float32_uint32 = matrix_fp32.view(np.uint32)
            # 提取高 16 位作为 bfloat16
            bf16_uint16 = (float32_uint32 >> 16).astype(np.uint16)
            # bfloat16 按照系统的本地字节序保存，无需字节交换
            matrix_converted = bf16_uint16
        elif data_format == "tfloat":
            # 将 float32 转换为 tfloat（清零尾数的低 13 位）
            tf32_matrix = matrix.copy()
            # 将 float32 数据视为 uint32 以进行位操作
            tf32_uint32 = tf32_matrix.view(np.uint32)
            # 掩码保留尾数的高 19 位
            tf32_uint32 &= 0xFFFFE000
            # 转换回 float32
            matrix_converted = tf32_uint32.view(np.float32)
        else:
            raise ValueError("Unsupported data format. Please choose: int8, half, bfloat16, tfloat.")

    # 将矩阵转换为字节
    output_bytes = matrix_converted.tobytes()

    # 如果未指定输出文件，生成唯一文件名
    if output_file is None:
        timestamp = int(time.time())
        output_file = f"matrix_{data_format}_{timestamp}.bin"

    # 将输出字节保存到二进制文件
    try:
        with open(output_file, 'wb') as f:
            f.write(output_bytes)
        logging.info(f"Matrix saved to {output_file} with format {data_format}.")
    except IOError as e:
        logging.error(f"Failed to write to file {output_file}: {e}")
        sys.exit(1)

def main():
    """
    主函数：解析命令行参数并生成矩阵。
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="生成一个矩阵并以二进制格式保存。")
    parser.add_argument("-m", type=int, required=True, help="矩阵的行数。")
    parser.add_argument("-n", type=int, required=True, help="矩阵的列数。")
    parser.add_argument(
        "-f", "--format", type=str, required=True,
        choices=["int8", "half", "bfloat16", "tfloat"],
        help="矩阵的数据格式（int8, half, bfloat16, tfloat）。"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出二进制文件的路径（默认：matrix_<format>_<timestamp>.bin）。"
    )

    args = parser.parse_args()

    # 验证输入参数
    if args.m <= 0 or args.n <= 0:
        logging.error("The number of rows and columns of the matrix must be positive integers.")
        sys.exit(1)

    # 生成矩阵
    generate_matrix(args.m, args.n, args.format, args.output)

if __name__ == "__main__":
    main()
