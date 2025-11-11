try:
    from msprobe.pytorch import PrecisionDebugger
    used_mstt = True
except ImportError:
    used_mstt = False
    print(f"!!!!!!!!!!!!!!!import mstt failed")

#如下示例dump指定代码块前反向数据。

"""Python
from transformers.utils import (
    PrecisionDebuggerINIT,
    PrecisionDebuggerMarkStep,
    PrecisionDebuggerBGN,
    PrecisionDebuggerEND,
)

# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
PrecisionDebuggerINIT(config_path="./config.json")

# 模型、损失函数的定义及初始化等操作

# 数据集迭代的位置一般为模型训练开始的位置
for data, label in data_loader:
	PrecisionDebuggerBGN() # 开启数据dump

	# 如下是模型每个step执行的逻辑
    output = model(data)
    PrecisionDebuggerEND() # 插入该函数到start函数之后，只dump start函数到该函数之间代码的前反向数据，本函数到stop函数之间的数据则不dump
    #...
    loss.backward()
    xm.mark_step()
	PrecisionDebuggerMarkStep() # 关闭数据dump,一定在mark_step()函数之后调用。
"""
def PrecisionDebuggerBGN():
    if used_mstt:
        PrecisionDebugger.start()
def PrecisionDebuggerEND():
    if used_mstt:
        PrecisionDebugger.forward_backward_dump_end()
#'/workspace/SPMD_TX8_DEVELOP/transformer/config_tensor.json'
def PrecisionDebuggerINIT( config_path,task=None,dump_path=None,level=None,model=None,step=None,):
    if used_mstt:
        return PrecisionDebugger(config_path=config_path,task=task,dump_path=dump_path,level=level,model=model,step=step)
    else:
        return None
def PrecisionDebuggerMarkStep():
    if used_mstt:
        PrecisionDebugger.stop()