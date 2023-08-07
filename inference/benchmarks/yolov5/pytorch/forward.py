from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync
from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync
from loguru import logger
from .dataloader import get_coco_api_from_dataset
from .utils import non_max_suppression, scale_boxes,index_list



def cal_perf(config, dataloader_len, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * dataloader_len * config.batch_size / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " ips")
    model_forward_core_perf = config.repeat * dataloader_len * config.batch_size / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " ips")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    start = time.time()
    core_time = 0.0
    acc = []

    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        all_top1 = []
        for step, (x, y) in enumerate(dataloader):
            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))

            with torch.no_grad():

                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                torch_sync(config)

                top1 = evaluator(pred, y)

                all_top1.extend(top1.cpu())
            core_time += time.time() - core_time_start
        acc.append(np.mean(all_top1))

    logger.info("Top1 Acc: " + str(acc))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)


def engine_forward(model, dataloader, evaluator, config):
    start = time.time()
    core_time = 0.0
    foo_time = 0.0
    acc = []
    results = {}
    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        coco = get_coco_api_from_dataset(dataloader.dataset)
        evaluator_instance = evaluator(coco)

        for step, (images, targets, im0) in enumerate(dataloader):

            torch_sync(config)
            core_time_start = time.time()

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                                str(len(dataloader)))
            with torch.no_grad():
                images = images.float()
                images /= 255  # 0 - 255 to 0.0 - 1.0
                outputs = model([images])
              
                pred = outputs[0][0]
                foo_time += outputs[1]
                pred = pred.float().cuda()

                output_shape = (1, 25200, 85)
                feat = pred.reshape(*output_shape)
                pred = feat.float()

                pred = non_max_suppression(pred, config.conf_thres, config.iou_thres, max_det=config.max_det)
                pred = torch.stack(pred).squeeze(0)

                pred[:,:4] = scale_boxes(images.shape[2:], pred[:, :4], im0[0].shape).round()

                results["boxes"] = pred[:,:4]
                results["scores"] = pred[:,4]

                label_new = [index_list[i] for i in pred[:,5].tolist()]

                label_new = torch.tensor(label_new, dtype=torch.int64)

                results["labels"] = label_new

                res = {
                    targets["image_id"].item(): results}
                    # for target, output in zip(targets, outputs)
                # }

                torch_sync(config)
                core_time += time.time() - core_time_start
                evaluator_instance.update(res)
            
        evaluator_instance.synchronize_between_processes()
        evaluator_instance.accumulate()
        evaluator_instance.summarize()
        ret = evaluator_instance.coco_eval['bbox'].stats.tolist()[0]

        logger.info("MAP: " + str(ret))

    duration = time.time() - start - foo_time
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(acc)), 3)
