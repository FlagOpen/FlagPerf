import os
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn


def create_model(args):
    resnet50_pretrained_path = os.path.join(args.data_dir,
                                            args.pretrained_path)
    coco_pretrained_path = os.path.join(args.data_dir,
                                        args.coco_weights_pretrained_path)
    url_resnet50_backbone = "file://" + resnet50_pretrained_path
    url_coco_pretrained = "file://" + coco_pretrained_path

    torchvision.models.resnet.__dict__['model_urls']['resnet50'] = url_resnet50_backbone
    torchvision.models.detection.mask_rcnn.__dict__['model_urls'][ 'maskrcnn_resnet50_fpn_coco'] = url_coco_pretrained
    model = maskrcnn_resnet50_fpn(pretrained=args.use_coco_pretrained)

    return model