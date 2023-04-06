import argparse


def parse_args_config():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = argparse.ArgumentParser(description="Config for model run cmd. ")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/mnt/dataset/",
                        help="Data directory.")
    args, unknown_args = parser.parse_known_args()
    return args