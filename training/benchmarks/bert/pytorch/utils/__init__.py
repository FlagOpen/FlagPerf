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

import inspect

from .check import check_config
from .dist import *


def is_property(value):
    status = [
        not callable(value), not inspect.isclass(value),
        not inspect.ismodule(value), not inspect.ismethod(value),
        not inspect.isfunction(value), not inspect.isbuiltin(value),
        not isinstance(value, classmethod)
    ]

    return all(status)
