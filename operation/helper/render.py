import re
import sys
import os
from jinja2 import Environment, FileSystemLoader

# Read TDP from environment variable
tdp = os.environ.get('TDP')
if tdp:
    single_card_tdp = tdp
else:
    single_card_tdp = ''

# Regular expressions for extracting desired values
# 用这个dict 从日志中提取数据
regex_dict = {
    # Core evaluation results
    'average_relative_error': r'Relative error with FP64-CPU: mean=(.*?),',
    'tflops': r'cputime=[0-9.]+\s+us,\s+throughput=[0-9.]+\s+op/s,\s+equals to (.*?) TFLOPS\s+',
    'kernel_clock': r'kerneltime=[0-9.]+\s+us,\s+throughput=[0-9.]+\s+op/s,\s+equals to (.*?) TFLOPS\s+',
    'fu_cputime': r'cputime=(.*?),',
    'kerneltime': r'FLOPS utilization: cputime=.*kerneltime=(.*?)\s+',
    # Other evaluation results
    #'relative_error': r' Relative error with FP64-CPU: mean=.*, std=(.*?)$',
    'relative_error': r'Relative error with FP64-CPU: mean=.*, std=([0-9.e-]+)\s+',
    'cpu_time': r'cputime=(.*?) us',
    'kernel_time': r'kerneltime=(.*?) us',
    'cpu_ops': r'cputime=.*, throughput=(.*?) op/s',
    'kernel_ops': r'kerneltime=.*, throughput=(.*?) op/s',
    'no_warmup_delay': r'no warmup=(.*?) us',
    'warmup_delay': r'no warmup=[0-9.]+\s+us,\s+warmup=(.*?) us',
    # Power monitoring results
    'ave_system_power': r'AVERAGE: (.*?) Watts',
    'max_system_power': r'MAX: (.*?) Watts',
    'system_power_stddev': r'STD DEVIATION: (.*?) Watts',
    'single_card_avg_power': r'RANK.* AVERAGE: (.*?) Watts',
    'single_card_max_power': r'RANK.* MAX: (.*?) Watts',
    'single_card_power_stddev': r'RANK.* STD DEVIATION: (.*?) Watts',
    # Other important monitoring results
    'avg_cpu_usage': r'SYSTEM CPU:\s+.*AVERAGE:\s+(\d+\.\d+)\s+%',
    'avg_mem_usage': r'SYSTEM MEMORY:\s+.*AVERAGE:\s+(\d+\.\d+)\s+%',
    'single_card_avg_temp': r'AI-chip TEMPERATURE:\s+.*AVERAGE: (.*?) °C',
    'max_gpu_memory_usage_per_card': r'AI-chip MEMORY:\s+.*AVERAGE:\s+\d+\.\d+ %,\s+MAX:\s+(\d+\.\d+) %',
}

# 用这个dict格式化生成最后的数据
format_dict = {
    # Core evaluation results
    'average_relative_error': ["2E"],
    'tflops': ["TFLOPS"],
    'kernel_clock': ["TFLOPS"],
    'fu_cputime': None,
    'kerneltime': None,
    # Other evaluation results
    #'relative_error': r' Relative error with FP64-CPU: mean=.*, std=(.*?)$',
    'relative_error': ["2E"],
    'cpu_time': ["us"],
    'kernel_time': ["us"],
    'cpu_ops': ["2F", 'op/s'],
    'kernel_ops': ["2F", 'op/s'],
    'no_warmup_delay': ['us'],
    'warmup_delay': ['us'],
    # Power monitoring results
    'ave_system_power': ['W'],
    'max_system_power': ['W'],
    'system_power_stddev': ['W'],
    'single_card_avg_power': ['W'],
    'single_card_max_power': ['W'],
    'single_card_power_stddev': ['W'],
    # Other important monitoring results
    'avg_cpu_usage': ['%'],
    'avg_mem_usage': ['%'],
    'single_card_avg_temp': ['°C'],
    'max_gpu_memory_usage_per_card': ['%'],
}

def read_log_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            log_text = file.read()
        return log_text
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None


def render(extracted_values, readme_file_path):
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    template_path = os.path.join(current_path)
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('template.md')
    rendered_text = template.render(extracted_values)
    dest_file_path = os.path.join(readme_file_path, "README.md")
    with open(dest_file_path, 'w') as file:
        file.write(rendered_text)

# Function to extract values using regular expressions
def extract_values_from_log(log_text, regex_dict):
    extracted_values = {}
    for key, regex_pattern in regex_dict.items():
        match = re.search(regex_pattern, log_text)
        if match:
            match_text = match.group(0)
            line_number = log_text[:match.start()].count('\n') + 1
            extracted_values[key] = match.group(1).strip()
        else:
            extracted_values[key] = None
    return extracted_values



def format_values(extracted_values, format_dict):
    formatted_values = {}
    for key, value in extracted_values.items():
        if key in format_dict:
            if format_dict[key]:
                for format_type in format_dict[key]:
                    if format_type == "2E":
                        value = float(value)
                        formatted_values[key] = f"{value:.2E}"
                    elif format_type == "2F":
                        value = float(value)
                        formatted_values[key] = str(round(value, 2))
                        print(formatted_values[key])
                    else:
                        formatted_values[key] = f"{value}{format_type}" if key not in formatted_values.keys() else f"{formatted_values[key]}{format_type}"
            else:
                formatted_values[key] = value
        else:
            formatted_values[key] = None
    return formatted_values

# Read log from file specified in command line argument
if __name__ == "__main__":
    if len(sys.argv) >=2 :
        file_name = sys.argv[1]
        data_type = sys.argv[2]
        readme_file_path = sys.argv[3]
        log_text = read_log_from_file(file_name)
        log_text = log_text.split("analysis logs")[1]
        if log_text:
            extracted_values = extract_values_from_log(log_text, regex_dict)
            for key, value in extracted_values.items():
                print(f"{key}: {value}")
            
            extracted_values = format_values(extracted_values, format_dict)
            extracted_values = {f"{data_type}_{key}": value for key, value in extracted_values.items()}
            extracted_values[f"{data_type}_single_card_tdp"] = single_card_tdp
            data_file = os.path.join(readme_file_path, "data.json")

            if os.path.exists(data_file):
                with open(data_file, 'r') as file:
                    data = file.read()
                # Merge the values from data file with extracted_values
                data_values = eval(data)
                extracted_values.update(data_values)
                with open(data_file, 'w') as file:
                    file.write(str(extracted_values))
                if len(extracted_values.keys()) >= 46:
                    render(extracted_values, readme_file_path)
            else:
                # Write extracted_values to data file
                with open(data_file, 'w+') as file:
                    file.write(str(extracted_values))
    else:
        print("Please provide a file name as a command line argument.")
