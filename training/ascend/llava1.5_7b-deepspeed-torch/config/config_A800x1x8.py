# ascend npu flashattention
import transformers
import os

cwd = os.getcwd()

os.chdir(os.path.dirname(__file__))

transformers_path = os.path.dirname(transformers.__file__)

import_utils_path = os.path.join(
    transformers_path, 
    "utils/import_utils.py"
)
modeling_llama_path = os.path.join(
    transformers_path, 
    "models/llama/modeling_llama.py"
)

import_utils_patch_bash = "patch --silent --forward " + \
	import_utils_path + \
	" import_utils.patch -o import_utils.py;"
modeling_llama_patch_bash = "patch --silent --forward " + \
	modeling_llama_path + \
	" modeling_llama.patch -o modeling_llama.py;"
    
if os.system(import_utils_patch_bash) == 0:
    os.system("mv import_utils.py " + import_utils_path + ";")
if os.system(modeling_llama_patch_bash) == 0:
    os.system("mv modeling_llama.py " + modeling_llama_path +";")
    
# useing torch_npu
os.system("cp ./train_mem.py ../../../benchmarks/llava1.5_7b/deepspeed-torch/train/")

os.chdir(cwd)


# Common arguments
theoryflops = 312000000000000.0

# pretrain arguments
pretrain_per_device_train_batch_size = 32
pretrain_gradient_accumulation_steps = 1


# finetune arguments
finetune_per_device_train_batch_size = 16
finetune_gradient_accumulation_steps = 1
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-7b"

# eval arguments
mmmu_data_path = "MMMU/MMMU"

os.chdir(cwd)
