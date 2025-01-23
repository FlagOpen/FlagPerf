import os

# from jinja2 import Environment, FileSystemLoader


# def render_base(readme_file_path, vendor, shm_size, chip):
#     v_chip = f'{vendor}_{chip}'
#     base_info = {"vendor": vendor, "v_chip": v_chip, "shm_size": shm_size, "chip": chip}
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     template_path = os.path.join(current_path)
#     env = Environment(loader=FileSystemLoader(template_path))
#     template = env.get_template('template.md')
#     rendered_text = template.render(base_info)
#     dest_file_path = os.path.join(readme_file_path, "README.md")
#     with open(dest_file_path, 'w') as file:
#         file.write(rendered_text)


def render(extracted_values, readme_file_path, vendor, shm_size, chip):
    json_data = []
    for key, value in extracted_values.items():
        json_data.append(value)
    dest_file_path = os.path.join(readme_file_path, "README.md")
    markdown_table = creat_markdown_table(json_data, vendor, shm_size, chip)
    with open(dest_file_path, 'w') as file:
        file.write(markdown_table)


def creat_markdown_table(data, vendor, shm_size, chip):
    v_chip = f'{vendor}_{chip}'
    table = f"# 参评AI芯片信息\n\n * 厂商：{vendor}\n * 产品名称：{v_chip}\n * 产品型号：{chip}\n * SHM_SIZE：{shm_size}\n\n\n\n"
    table += "| op_name | dtype | shape_detail | 无预热时延(Latency-No warmup) | 预热时延(Latency-Warmup) | 原始吞吐(Raw-Throughput)| 核心吞吐(Core-Throughput) | 实际算力开销 | 实际算力利用率 | 实际算力开销(内核时间) | 实际算力利用率(内核时间) |\n| --- | ---| --- | ---| --- | ---| --- | ---| --- | ---| --- |\n"
    for row in data:
        table += f"| {row['op_name']} | {row['dtype']} | {row['shape_detail']} | {row['no_warmup_latency']} | {row['warmup_latency']} | {row['raw_throughput']} | {row['core_throughput']} | {row['ctflops']} | {row['cfu']} | {row['ktflops']} | {row['kfu']} |\n"
    return table