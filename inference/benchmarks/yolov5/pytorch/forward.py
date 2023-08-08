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
from .utils import non_max_suppression, scale_boxes, index_list



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


def engine_forward(model, dataloader, evaluator, config):
    start = time.time()
    core_time = 0.0
    foo_time = 0.0
    result = {}
    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))

        coco = get_coco_api_from_dataset(dataloader.dataset)
        evaluator_instance = evaluator(coco)
        for step, (images, targets, im0) in enumerate(dataloader):

            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                                str(len(dataloader)))
            with torch.no_grad():
                images = [torch.from_numpy(image) for image in images]
                images = torch.stack(images, dim=0)
                images = images.float()
                images /= 255  # 0 - 255 to 0.0 - 1.0

                torch_sync(config)
                core_time_start = time.time()

                outputs = model([images])

                pred = outputs[0]
                foo_time += outputs[1]
                torch_sync(config)
                core_time += time.time() - core_time_start
                pred = pred[0].float().cpu()

                output_shape = (config.batch_size, config.hidden_size, config.number_boxes)
                batch_pred = pred.reshape(*output_shape)
                for index in range(batch_pred.shape[0]):
                    pred = batch_pred[index].unsqueeze(0)
                    pred = non_max_suppression(pred, config.conf_thres, config.iou_thres, max_det=config.max_det)
                    pred = torch.stack(pred).squeeze(0)
                    pred[:,:4] = scale_boxes(images.shape[2:], pred[:, :4], im0[index].shape).round()
                    result["boxes"] = pred[:,:4]
                    result["scores"] = pred[:,4]
                    new_pred = [index_list[i] for i in pred[:,5].tolist()]
                    new_pred = torch.tensor(new_pred, dtype=torch.int64)
                    result["labels"] = new_pred
                    res = {
                        targets[index]["image_id"]: result
                    }
                    evaluator_instance.update(res)

        evaluator_instance.synchronize_between_processes()
        evaluator_instance.accumulate()
        evaluator_instance.summarize()
        ret = evaluator_instance.coco_eval['bbox'].stats.tolist()[0]

        logger.info("MAP: " + str(ret))

    duration = time.time() - start - foo_time
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time - foo_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(ret, 3)
