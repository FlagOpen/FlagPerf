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

import numpy as np


from .distributed import DistributedDlrm
from utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed, get_rank

from dataloaders.utils import prefetcher, get_embedding_sizes

def create_model(args, device, device_mapping, feature_spec):
    rank = get_rank()
    bottom_mlp_sizes = args.bottom_mlp_sizes if rank == device_mapping['bottom_mlp'] else None
    world_embedding_sizes = get_embedding_sizes(feature_spec, max_table_size=args.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
     # Embedding sizes for each GPU
    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping['embedding'][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=args.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=args.top_mlp_sizes,
        interaction_op=args.interaction_op,
        fp16=args.amp,
        use_cpp_mlp=args.optimized_mlp,
        bottom_features_ordered=args.bottom_features_ordered,
        device=device)
    return model
