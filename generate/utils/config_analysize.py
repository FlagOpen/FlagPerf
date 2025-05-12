import yaml
from collections import deque
import re


def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def read_last_n_lines(file_path, n=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = deque(file, n)
        lines = list(lines)
    return lines
