# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper


from train import trainer
from train.training_state import TrainingState

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config
    
    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())  # _base.py增加模型名称name
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time # init起始时间，单位ms

    init_helper.set_seed(config.seed, config.vendor)


    # 创建TrainingState对象
    training_state = TrainingState()

    if not config.do_train:
        return config, training_state

    # init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time # init结束时间，单位为ms
    training_state.init_time = (init_end_time - init_start_time) / 1e+3 # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time # 训练起始时间，单位为ms

    # 训练过程
    
    trainer.run(config, training_state)
    
    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)
    
    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:
        
            
        training_perf = (dist_pytorch.global_batch_size(config_update) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_imgs_per_second": training_perf,
            "converged": state.converged,
            "final_P": state.P,
            "final_R": state.R,
            "final_mAP50": state.mAP50,
            "final_mAP": state.mAP,
            "final_fitness": state.best_fitness,   
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
