import argparse
import os
import re
import stat
from typing import Tuple
from os import path as osp

import matplotlib.pyplot as plt


def plot(args):
    assert args.in_file, "in_file can NOT be empty."
    out_dir = args.out_dir
    out_suffix = args.image_extension if not args.out_file else get_ext(
        args.out_file)

    eval_acc = []
    global_steps = []

    # 根据日志中metric格式，修改正则
    patt_eval_acc1 = '"eval_acc1": (.*?),'  # non-greedy RegExp
    patt_global_steps = '"global_steps": (.*?),'  # non-greedy RegExp

    # 读取日志文件
    with open(args.in_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "EVALUATE" in line:
                eval_acc_val = extract_metric(line, patt_eval_acc1)
                global_steps_val = extract_metric(line, patt_global_steps)

                if idx % 100 == 0:
                    # 注意这里要将str转成数值类型，否则图像上的label过于密集
                    eval_acc.append(float(eval_acc_val))
                    global_steps.append(int(global_steps_val))

    plt.plot(global_steps, eval_acc)
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)

    fn_without_ext, _ = split_filename(args.in_file)
    out_fn = get_basename(fn_without_ext) + out_suffix

    if hasattr(args, "out_file") and args.out_file:
        out_fn = get_basename(args.out_file)

    out_file = osp.join(out_dir, out_fn)

    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # 设置读写权限
        os.chmod(out_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)

    print(f"image save successfully. path:{out_file}")
    plt.savefig(out_file)

    if args.show_image:
        os.system(f"open {out_file}")


def split_filename(file_path) -> Tuple:
    """
    get file basename
    /path/to/filename.ext => filename, extension
    """
    file_basename = osp.basename(file_path)
    # splitext返回一个元组，第一个元素是文件名（不含后缀），第二个元素是文件的后缀（含点号）
    basename, file_extension = osp.splitext(file_basename)
    return basename, file_extension


def get_ext(file_path):
    _, ext = split_filename(file_path)
    return ext


def get_basename(file_path):
    file_basename = osp.basename(file_path)
    return file_basename


def extract_metric(line, patt) -> str:
    res = re.findall(patt, line)
    return str(res[0]) if res else ""


def get_args_parser():
    parser = argparse.ArgumentParser(description="argparser for plot")
    # file related
    parser.add_argument("-i",
                        "--in_file",
                        default=None,
                        help="fullpath of input log file")
    parser.add_argument("-o",
                        "--out_file",
                        default=None,
                        help="fullpath of output file")
    parser.add_argument("-d", "--out_dir", default="/tmp")
    parser.add_argument("-e",
                        "--image_extension",
                        default=".png",
                        help="extension of output file")

    # plot related
    parser.add_argument("-x", "--xlabel", default="Global Steps")
    parser.add_argument("-y", "--ylabel", default="Accuracy")
    parser.add_argument("-t", "--title", default="ResNet50 accuracy")
    parser.add_argument("-s", "--show-image", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_parser()
    plot(args)
