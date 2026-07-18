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

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CrossEntropyCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.padding_idx = args.padding_idx

    def forward(self, norm_probs, target, reduce=True):
        """Compute the loss for the given sample.
        """
        lprobs = norm_probs.view(-1, norm_probs.size(-1))
        target = target.view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        return loss


class LabelSmoothedCrossEntropyCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.eps = args.label_smoothing
        self.padding_idx = args.padding_idx

    def forward(self, norm_probs, target, reduce=True):
        """Compute the loss for the given sample.
        """
        target = target.view(-1, 1)
        lprobs = norm_probs.view(-1, norm_probs.size(-1))
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        return loss


CRITERION_REGISTRY = {
        'label_smoothed_cross_entropy' : LabelSmoothedCrossEntropyCriterion,
        'cross_entropy' : CrossEntropyCriterion,
        }
