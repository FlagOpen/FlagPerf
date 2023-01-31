import torch.nn as nn
from typing import Any

class Loss(object):

    def get_criterion(device)->Any:
        criterion = nn.CrossEntropyLoss().to(device)
        return criterion