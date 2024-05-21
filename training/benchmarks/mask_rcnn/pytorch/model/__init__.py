# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def create_model(args):
    resnet50_pretrained_path = os.path.join(args.data_dir,
                                            args.pretrained_path)
    url_resnet50_backbone = "file://" + resnet50_pretrained_path
    torchvision.models.resnet.__dict__['model_urls']['resnet50'] = url_resnet50_backbone
    model = maskrcnn_resnet50_fpn(pretrained=False)
    return model