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

# Returns true only if resuming from a checkpoint found in output_dir.
# init_checkpoint and init_tf_checkpoint are not considered
import glob
import os
from collections import OrderedDict


def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return args.resume_from_checkpoint and (
        args.resume_init_checkpoint is not None
        or len(glob.glob(os.path.join(args.output_dir, checkpoint_str)))) > 0


def remap_attn_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'self.query.weight' in k:
                new_k = k.replace('self.query.weight',
                                  'multi_head_attention.q_weight')
            elif 'self.key.weight' in k:
                new_k = k.replace('self.key.weight',
                                  'multi_head_attention.k_weight')
            elif 'self.value.weight' in k:
                new_k = k.replace('self.value.weight',
                                  'multi_head_attention.v_weight')
            elif 'self.query.bias' in k:
                new_k = k.replace('self.query.bias',
                                  'multi_head_attention.q_bias')
            elif 'self.key.bias' in k:
                new_k = k.replace('self.key.bias',
                                  'multi_head_attention.k_bias')
            elif 'self.value.bias' in k:
                new_k = k.replace('self.value.bias',
                                  'multi_head_attention.v_bias')
            elif 'output.dense.weight' in k:
                new_k = k.replace('output.dense.weight',
                                  'multi_head_attention.out_proj_weight')
            elif 'output.dense.bias' in k:
                new_k = k.replace('output.dense.bias',
                                  'multi_head_attention.out_proj_bias')
            elif 'output.LayerNorm.weight' in k:
                new_k = k.replace('output.LayerNorm.weight',
                                  'layer_norm.weight')
            elif 'output.LayerNorm.bias' in k:
                new_k = k.replace('output.LayerNorm.bias', 'layer_norm.bias')
            else:
                new_k = k
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict


def remap_segmented_model_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'bert' in k:
            new_k = 'bert_model_segment.' + k
        elif 'cls' in k:
            new_k = 'heads_only_segment.' + k
        else:
            assert False, "shouldn't happen"
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict
