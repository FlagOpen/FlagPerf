# Copyright 2026 FlagOS Contributors
#
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

class Backend(object):
    NAME: str = ""

    def __init__(self) -> None:
        pass

    def synchronize(self):
        raise "Not implemented."

    def set_device(self, local_rank):
        raise "Not implemented."


class CudaBackend(Backend):
    NAME = 'CUDA'

    def __init__(self) -> None:
        super().__init__()

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def set_device(self, rank):
        import torch
        torch.cuda.set_device(rank)
