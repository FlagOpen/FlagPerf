import numpy as np
import argparse
import sys
import os
import math

def convert_bfloat16_to_float32(bf16_array):
    """
    Convert a NumPy array of bfloat16 stored as uint16 to float32.

    Parameters:
    - bf16_array (np.ndarray): A uint16 array representing bfloat16.

    Returns:
    - np.ndarray: The converted float32 array.
    """
    # Ensure the input array is of type uint16
    if bf16_array.dtype != np.uint16:
        raise TypeError("Input array must be of type uint16.")

    # Assume data is in native byte order without conversion
    bf16 = bf16_array

    # Shift bfloat16 left by 16 bits, padding the lower 16 bits with 0 to convert to float32
    float32_bits = bf16.astype(np.uint32) << 16

    # Interpret the bit pattern as float32
    float32_array = float32_bits.view(np.float32)

    return float32_array

def read_bin(file_path, data_format, total_rows, total_cols, rb, re, cb, ce):
    """
    Read a binary file, reshape it into a matrix, and extract a submatrix based on indices.

    Parameters:
    - file_path (str): Path to the binary file.
    - data_format (str): Data format ('int8', 'half', 'bfloat16', 'tfloat').
    - total_rows (int): Total number of rows in the matrix.
    - total_cols (int): Total number of columns in the matrix.
    - rb (int): Starting row index.
    - re (int): Ending row index (exclusive).
    - cb (int): Starting column index.
    - ce (int): Ending column index (exclusive).

    Returns:
    - np.ndarray: The extracted submatrix.
    """
    # Define data types and bytes per element based on data_format
    data_type_map = {
        "int8": (np.int8, 1),
        "half": (np.float16, 2),
        "bfloat16": (np.uint16, 2),  # Use unsigned 16-bit integers to store bfloat16
        "tfloat": (np.float32, 4)
    }

    if data_format not in data_type_map:
        raise ValueError("Unsupported data format. Please choose from: int8, half, bfloat16, tfloat.")

    dtype, bytes_per_element = data_type_map[data_format]

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: File does not exist - {file_path}", file=sys.stderr)
        sys.exit(1)

    # Get the file size in bytes
    total_bytes = os.path.getsize(file_path)
    if total_bytes == 0:
        print(f"Error: File is empty - {file_path}", file=sys.stderr)
        sys.exit(1)

    # Calculate the total number of elements in the file
    total_elements = total_bytes // bytes_per_element

    # Calculate the required total number of elements
    requested_elements = total_rows * total_cols

    if requested_elements > total_elements:
        print(f"Error: Matrix size ({total_rows}x{total_cols}) exceeds the data available in the file.", file=sys.stderr)
        sys.exit(1)

    # Read the necessary amount of data
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(requested_elements * bytes_per_element)
    except IOError as e:
        print(f"Error: Cannot open file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Process the data based on data_format
    try:
        if data_format in ["int8", "half", "tfloat"]:
            data = np.frombuffer(raw_data, dtype=dtype)
        elif data_format == "bfloat16":
            bf16 = np.frombuffer(raw_data, dtype=dtype)
            data = convert_bfloat16_to_float32(bf16)
    except Exception as e:
        print(f"Error: Failed to process data in format {data_format}: {e}", file=sys.stderr)
        sys.exit(1)

    # Reshape into a matrix
    try:
        matrix = data.reshape(total_rows, total_cols, order='C')
    except ValueError as e:
        print(f"Error: Cannot reshape data into a matrix of size ({total_rows}x{total_cols}): {e}", file=sys.stderr)
        sys.exit(1)

    # Handle row and column indices
    # Default values: extract the entire matrix
    if rb is None:
        rb = 0
    if re is None:
        re = total_rows
    if cb is None:
        cb = 0
    if ce is None:
        ce = total_cols

    # Validate index ranges
    if not (0 <= rb < total_rows) or not (0 < re <= total_rows) or rb >= re:
        print("Error: Invalid row index range.", file=sys.stderr)
        sys.exit(1)
    if not (0 <= cb < total_cols) or not (0 < ce <= total_cols) or cb >= ce:
        print("Error: Invalid column index range.", file=sys.stderr)
        sys.exit(1)

    # Extract the submatrix
    sub_matrix = matrix[rb:re, cb:ce]

    return sub_matrix

def main():
    parser = argparse.ArgumentParser(description="Read a binary file and display its contents as a matrix.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the binary file.")
    parser.add_argument("-r", type=int, required=True, help="Total number of rows in the matrix.")
    parser.add_argument("-c", type=int, required=True, help="Total number of columns in the matrix.")
    parser.add_argument(
        "-d", "--data_format", type=str, required=True,
        choices=["int8", "half", "bfloat16", "tfloat"],
        help="Data format of the matrix (int8, half, bfloat16, tfloat)."
    )
    parser.add_argument("-rb", type=int, default=None, help="Starting row index (0-based).")
    parser.add_argument("-re", type=int, default=None, help="Ending row index (exclusive).")
    parser.add_argument("-cb", type=int, default=None, help="Starting column index (0-based).")
    parser.add_argument("-ce", type=int, default=None, help="Ending column index (exclusive).")

    args = parser.parse_args()

    # Validate input parameters
    if args.r <= 0 or args.c <= 0:
        print("Error: Total number of rows and columns must be positive integers.", file=sys.stderr)
        sys.exit(1)
    if args.rb is not None and args.rb < 0:
        print("Error: Starting row index must be a non-negative integer.", file=sys.stderr)
        sys.exit(1)
    if args.re is not None and args.re <= 0:
        print("Error: Ending row index must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if args.cb is not None and args.cb < 0:
        print("Error: Starting column index must be a non-negative integer.", file=sys.stderr)
        sys.exit(1)
    if args.ce is not None and args.ce <= 0:
        print("Error: Ending column index must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    # Read and reshape the matrix
    matrix = read_bin(
        args.file,
        args.data_format,
        args.r,
        args.c,
        args.rb,
        args.re,
        args.cb,
        args.ce
    )

    # Set NumPy to print all elements without summarization
    np.set_printoptions(threshold=sys.maxsize)

    # print("Matrix content:")
    print(matrix)

if __name__ == "__main__":
    main()
