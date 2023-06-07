import os

# eg. in_chans=config.MODEL.SWIN.IN_CHANS,

def trans_model(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            if isinstance(line,str) and "config" in line and "=" in line:
                after_tran_str = line.split(".",1)[1].lower().replace(".","_").strip()
                print(line.split(".",1)[0] + "." + after_tran_str)
            else:
                print(line)


# _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]              
def trans_conf(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if line.startswith("_C") and "." in line:
                new_conf = line.split(".",1)[1].lower().replace(".","_").strip()
                print(new_conf)
            else:
                print(line.replace("\n",""))

if __name__ == "__main__":
    # model_file_name = "/data/sen.li/workspace/code/FlagPerf/training/benchmarks/swin_transformer/pytorch/models/build.py"
    # trans_model(model_file_name)
    conf_file_name = "/data/sen.li/workspace/code/FlagPerf/training/benchmarks/swin_transformer/pytorch/config/swin_config.py"
    trans_conf(conf_file_name)