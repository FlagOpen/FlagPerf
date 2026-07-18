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

import torch


def cross_entropy(outputs, target):
    """
    Compute the cross entropy loss of output and target. 

    para:   outputs, [b, s, vocab_size]
            target, [b, s]
    return: loss, [b, s]
    """

    logits = outputs.clone()
    # logits = outputs
    logits_max = torch.max(logits, dim=-1)[0]

    # Subtract the maximum value.
    logits.sub_(logits_max.unsqueeze(dim=-1))
    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    logits_2d = logits.view(-1, logits.size()[-1])
    target_1d = target.view(-1)
    arange_1d = torch.arange(start=0,
                             end=logits_2d.size()[0],
                             device=logits_2d.device)
    predit_ligits_1d = logits_2d[arange_1d, target_1d]
    predit_ligits = predit_ligits_1d.view_as(target)

    loss = torch.log(sum_exp_logits) - predit_ligits
    return loss
