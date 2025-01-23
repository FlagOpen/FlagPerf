import numpy as np
import argparse
import os
import sys

def get_endian_prefix():
    if sys.byteorder == 'little':
        return '<'
    else:
        return '>'

def read_matrix(file_path, data_format, rows, cols):
    """
    Read matrix data from a binary file and convert it to a float32 NumPy array.

    Parameters:
    - file_path (str): Path to the binary file.
    - data_format (str): Data format, supports 'int8', 'half', 'bfloat16', 'tfloat'.
    - rows (int): Number of rows in the matrix.
    - cols (int): Number of columns in the matrix.

    Returns:
    - np.ndarray: Converted float32 matrix with shape (rows, cols).
    """
    # Define NumPy data types and bytes per element for each data format
    format_specs = {
        "int8": (np.int8, 1),
        "half": (np.float16, 2),
        "bfloat16": (np.uint16, 2),  # Read as unsigned 16-bit integers
        "tfloat": (np.float32, 4)
    }

    if data_format not in format_specs:
        raise ValueError("Unsupported data format. Please choose: int8, half, bfloat16, tfloat.")

    dtype, bytes_per_element = format_specs[data_format]
    expected_bytes = rows * cols * bytes_per_element
    actual_bytes = os.path.getsize(file_path)

    # Check if the file size matches the expected size
    if actual_bytes < expected_bytes:
        raise ValueError(f"File {file_path} size is smaller than expected. Expected {expected_bytes} bytes, got {actual_bytes} bytes.")
    elif actual_bytes > expected_bytes:
        print(f"Warning: File {file_path} contains more data than expected. Extra data will be ignored.", file=sys.stderr)

    # Read the expected number of bytes
    with open(file_path, 'rb') as f:
        raw_data = f.read(expected_bytes)

    endian = get_endian_prefix()

    # Convert to float32 based on data format
    if data_format == "int8":
        data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
    elif data_format == "half":
        half_dtype = np.dtype(endian + 'f2')  # 'f2' represents float16
        data = np.frombuffer(raw_data, dtype=half_dtype).astype(np.float32)
    elif data_format == "bfloat16":
        # Read bfloat16 data with correct byte order
        bf16_dtype = np.dtype(endian + 'u2')  # 'u2' represents unsigned 16-bit integer
        bf16 = np.frombuffer(raw_data, dtype=bf16_dtype)
        # Convert bfloat16 to float32
        float32_bits = bf16.astype(np.uint32) << 16
        data = float32_bits.view(np.float32)
    elif data_format == "tfloat":
        # Read as float32
        data = np.frombuffer(raw_data, dtype=np.float32)
        # Clear the lower 13 bits of the mantissa to convert to tfloat format
        # tf32 = data.copy()
        # tf32_uint32 = tf32.view(np.uint32)
        # tf32_uint32 &= 0xFFFFE000  # Keep the top 19 bits
        # data = tf32_uint32.view(np.float32)
    else:
        raise ValueError("Unsupported data format. Please choose: int8, half, bfloat16, tfloat.")

    return data.reshape((rows, cols))

def write_matrix(matrix, data_format, output_file):
    """
    Convert a float32 NumPy matrix to the specified data format and write it to a binary file.

    Parameters:
    - matrix (np.ndarray): Matrix in float32 format.
    - data_format (str): Output data format, supports 'int8', 'half', 'bfloat16', 'tfloat'.
    - output_file (str): Path to the output file.
    """
    endian = get_endian_prefix()

    # Define processing for each data format
    if data_format == "int8":
        # Clip data to int8 range and convert to int8 type
        matrix_clipped = np.clip(matrix, -128, 127)
        matrix_converted = np.round(matrix_clipped).astype(np.int8)
        output_bytes = matrix_converted.tobytes()
    elif data_format == "half":
        # Convert float32 to float16 with system's native byte order
        matrix_converted = matrix.astype(np.float16).newbyteorder(endian)
        output_bytes = matrix_converted.tobytes()
    elif data_format == "bfloat16":
        # Convert float32 to bfloat16
        float32_uint32 = matrix.view(np.uint32)
        bf16_uint16 = (float32_uint32 >> 16).astype(np.uint16)
        # Apply correct byte order
        bf16_uint16 = bf16_uint16.newbyteorder(endian)
        output_bytes = bf16_uint16.tobytes()
    elif data_format == "tfloat":
        # Convert float32 to tfloat (clear lower 13 bits)
        tf32 = matrix.copy()
        tf32_uint32 = tf32.view(np.uint32)
        tf32_uint32 &= 0xFFFFE000  # Keep the top 19 bits
        tf32 = tf32_uint32.view(np.float32)
        output_bytes = tf32.tobytes()
    else:
        raise ValueError("Unsupported data format. Please choose: int8, half, bfloat16, tfloat.")

    # Write the converted data to the binary file
    with open(output_file, 'wb') as f:
        f.write(output_bytes)

def perform_gemm(m, n, k, left_file, right_file, di, do, output_file):
    """
    Perform General Matrix-Matrix Multiplication (GEMM): C = A x B, and save the result.

    Parameters:
    - m (int): Number of rows in the left matrix A.
    - n (int): Number of columns in the right matrix B.
    - k (int): Number of columns in A and rows in B.
    - left_file (str): Path to the left matrix A binary file.
    - right_file (str): Path to the right matrix B binary file.
    - di (str): Input data type, supports 'int8', 'half', 'bfloat16', 'tfloat'.
    - do (str): Output data type, supports 'int8', 'half', 'bfloat16', 'tfloat'.
    - output_file (str): Path to save the result.
    """
    print("Reading left matrix A...")
    A = read_matrix(left_file, di, m, k)
    print(f"Shape of left matrix A: {A.shape}")

    print("Reading right matrix B...")
    B = read_matrix(right_file, di, k, n)
    print(f"Shape of right matrix B: {B.shape}")

    print("Performing matrix multiplication (GEMM)...")
    C = np.matmul(A, B)
    print(f"Shape of result matrix C: {C.shape}")

    print("Converting and saving the output matrix...")
    write_matrix(C, do, output_file)
    print(f"Result matrix saved to {output_file} with data type {do.upper()}.")

def main():
    """
    Main function: Parse command-line arguments and perform GEMM operation.
    """
    parser = argparse.ArgumentParser(description="Perform GEMM (General Matrix-Matrix Multiplication).")
    parser.add_argument("-m", type=int, required=True, help="Number of rows in the left matrix.")
    parser.add_argument("-n", type=int, required=True, help="Number of columns in the right matrix.")
    parser.add_argument("-k", type=int, required=True, help="Number of columns in the left matrix and rows in the right matrix.")
    parser.add_argument("-L", type=str, required=True, help="Path to the left matrix A binary file.")
    parser.add_argument("-R", type=str, required=True, help="Path to the right matrix B binary file.")
    parser.add_argument("-di", type=str, required=True, choices=["int8", "half", "bfloat16", "tfloat"],
                        help="Input data type ('int8', 'half', 'bfloat16', 'tfloat').")
    parser.add_argument("-do", type=str, required=True, choices=["int8", "half", "bfloat16", "tfloat"],
                        help="Output data type ('int8', 'half', 'bfloat16', 'tfloat').")
    parser.add_argument("-o", type=str, required=True, help="Path to save the result.")

    args = parser.parse_args()

    # Verify that matrix dimensions are positive integers
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        print("Error: Matrix dimensions m, n, k must be positive integers.", file=sys.stderr)
        sys.exit(1)

    # Check if input files exist
    if not os.path.isfile(args.L):
        print(f"Error: Left matrix file does not exist - {args.L}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.R):
        print(f"Error: Right matrix file does not exist - {args.R}", file=sys.stderr)
        sys.exit(1)

    def check_file_size(file_path, data_format, rows, cols):
        """
        Check if the file size matches the expected matrix dimensions.

        Parameters:
        - file_path (str): File path.
        - data_format (str): Data format.
        - rows (int): Number of rows in the matrix.
        - cols (int): Number of columns in the matrix.
        """
        format_specs = {
            "int8": 1,
            "half": 2,
            "bfloat16": 2,
            "tfloat": 4
        }

        if data_format not in format_specs:
            raise ValueError("Unsupported data format. Please choose: int8, half, bfloat16, tfloat.")

        bytes_per_element = format_specs[data_format]
        expected_bytes = rows * cols * bytes_per_element
        actual_bytes = os.path.getsize(file_path)

        if actual_bytes < expected_bytes:
            raise ValueError(f"File {file_path} size is smaller than expected. Expected {expected_bytes} bytes, got {actual_bytes} bytes.")
        elif actual_bytes > expected_bytes:
            print(f"Warning: File {file_path} contains more data than expected. Extra data will be ignored.", file=sys.stderr)

    # Verify that input files have the correct size
    try:
        check_file_size(args.L, args.di, args.m, args.k)
        check_file_size(args.R, args.di, args.k, args.n)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform GEMM operation
    perform_gemm(args.m, args.n, args.k, args.L, args.R, args.di, args.do, args.o)

if __name__ == "__main__":
    main()
