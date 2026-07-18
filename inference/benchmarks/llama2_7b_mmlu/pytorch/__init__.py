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

from .dataloader import build_dataloader
from .model import create_model
from .export import export_model
from .evaluator import evaluator
from .forward import model_forward, engine_forward
import os 

env = os.environ['vendor']

if env == "hexaflake":
    from .hexaflake import hx_dataloader
    build_dataloader = hx_dataloader

    from .hexaflake import hx_model
    create_model = hx_model

    from .hexaflake import hx_export_model
    export_model = hx_export_model

    from .hexaflake import hx_evaluator
    evaluator = hx_evaluator

    from .hexaflake import hx_model_forward, hx_engine_forward
    model_forward = hx_model_forward
    engine_forward = hx_engine_forward
