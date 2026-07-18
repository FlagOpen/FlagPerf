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

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
from packaging.version import Version

def create_model():
    TORCHVISION_VERSION = Version(torchvision.__version__).base_version
    if Version(TORCHVISION_VERSION) > Version("0.12.0"):
        # model_urls has gone since v0.13.0
        torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1.value.url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    else:
        torchvision.models.resnet.__dict__['model_urls'][
            'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    return fasterrcnn_resnet50_fpn()
