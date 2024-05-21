#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import sys
import os


this_dir = os.path.dirname(os.path.abspath(__file__))


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')


batch_utils = CppExtension(
                        name='fairseq_data.batch_C',
                        sources=[os.path.join(this_dir, 'make_batches.cpp')],
                        extra_compile_args={
                                'cxx': ['-O2',],
                        }
)
setup(
    name='fairseq_data',
    version='0.5.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    packages=find_packages(),
    ext_modules=[batch_utils],
    cmdclass={
                'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    test_suite='tests',
)
