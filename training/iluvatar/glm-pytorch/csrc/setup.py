import os
import torch
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension
import subprocess
from torch.utils import cpp_extension
from setuptools import setup

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

version = '0.1'

cmdclass = {}
cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=True)

if torch.utils.cpp_extension.CUDA_HOME is None:
    raise RuntimeError("--fast_multihead_attn was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")

this_dir = os.path.dirname(os.path.abspath(__file__))

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']

version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

COMMON_COREX_FLAGS = ["--cuda-gpu-arch=ivcore10", "--cuda-path=" + cpp_extension.CUDA_HOME]

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, 'include', 'ATen', 'CUDAGenerator.h')):
    generator_flag = ['-DOLD_GENERATOR']

cc_flag = []
_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=sm_80')

ext_modules = []
ext_modules.append(
    CUDAExtension(name='glm_fast_self_multihead_attn_bias',
                  sources=[os.path.join(this_dir, 'self_multihead_attn_bias.cpp'),
                           os.path.join(this_dir, 'self_multihead_attn_bias_cuda.cu')],
                  extra_compile_args={'cxx': ['-O3',] + version_dependent_macros + generator_flag,
                                              'nvcc':['-O3',
                                                      #'-gencode', 'arch=compute_70,code=sm_70',
                                                      '-I./apex/contrib/csrc/multihead_attn/cutlass/',
                                                      '-U__CUDA_NO_HALF_OPERATORS__',
                                                      '-U__CUDA_NO_HALF_CONVERSIONS__',
                                                      #'--expt-relaxed-constexpr',
                                                      #'--expt-extended-lambda',
                                                      #'--use_fast_math'
                                                     ] + version_dependent_macros + generator_flag + cc_flag + COMMON_COREX_FLAGS},
                  ))
setup(
    name='apex',
    version=version,
    description='PyTorch Extensions for glm self attention',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=False,
    zip_safe=False
)