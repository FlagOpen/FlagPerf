import torch
import numpy as np


def evaluator(pred, gt_path):
    batch_size = len(pred)

    result = []
    for i in range(batch_size):
        mask = pred[i][0]

        num_mask = len(mask)

        for j in range(num_mask):
            mask_img = mask[j]

            gt = torch.load(gt_path + '_' + str(j) + ".pt")
            gt = gt.to(mask_img.device)
            iou = torch.eq(mask_img, gt).sum().item() / mask_img.numel()
            result.append(float(iou))

    return np.mean(result)
