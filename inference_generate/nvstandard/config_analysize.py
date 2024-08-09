import yaml
from collections import deque
import re
def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data
def read_last_n_lines(file_path, n=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 使用 deque 来存储文件的最后 n 行
        lines = deque(file, n)
        lines =list(lines)
    return lines
# data=read_last_n_lines('/home/daliwang/FlagPerf/newinference/model/nvstandard/task_time.log',8)
# for i in data:
#     if "All Flops" in i:
#         numbers=re.findall(r'\d+', i)
#         print(numbers)
# 使用示例
# yaml_data = read_yaml_file('/home/daliwang/FlagPerf/newinference/model/config/GPU_config.yaml')
# print(yaml_data)
# print(type(yaml_data))
