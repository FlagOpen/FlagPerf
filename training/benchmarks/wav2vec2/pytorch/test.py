import sys
import os
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from itertools import cycle, islice
from dataloaders.dataloader_wav import build_train_dataloader
from dataloaders.dataset import build_train_dataset, build_eval_dataset
from train.trainer import Trainer
from train import trainer_adapter



"""Pytorch Pretraining Example"""
"""
说明：文档中所有TODO的地方，都需要自定义实现。尽量保证接口一致。没有标记TODO的地方，可以参考示例中的实现，或者在此基础上做些微调。
"""

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
from dataloaders.dataset import build_train_dataset, build_eval_dataset
from itertools import cycle, islice
from dataloaders.dataloader_wav import build_train_dataloader#, build_eval_dataloader

def main() -> Tuple[Any, Any]:



    global config
    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())

    dist_pytorch.init_dist_training_env(model_driver.config)
    # dist_pytorch.barrier(config.vendor)


    init_helper.set_seed(config.seed + config.local_rank, model_driver.config.vendor)

    train_dataset = build_train_dataset(config.train_subset, config,
                                 with_labels=False, training=True)
    
    train_dataloader, sampler= build_train_dataloader(
        train_dataset,
        True,
        max_tokens=config.max_tokens,
        max_sentences=config.batch_size,
        max_positions=(config.max_tokens, config.max_tokens),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=config.required_batch_size_multiple,
        seed=config.seed,
        num_shards=2,
        shard_id=int(os.getenv("RANK")),
        num_workers=config.num_workers,
        num_concat_batches=config.num_concat_batches)

    # print("111")
    # itr = islice(train_dataloader,1388)
    # print(itr,next(itr))

    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=None,
                      training_state=None,
                      device=config.device,
                      config=config)
    trainer.init()
    trainer.train_one_epoch(train_dataloader,sampler)

    
if __name__=="__main__":
    print("main",os.environ.get('WORLD_SIZE'))

    main()

'''
python run.py
python3 -m torch.distributed.launch training/benchmarks/wav2vec2/pytorch/test.py --extern_config_dir /workspace/wav2vec2/wav2vec2_Perf/training/nvidia/wav2vec2-pytorch/config --extern_config_file config_A100x1x8.py --data_dir /workspace/wav2vec2/wav2vec2_data/LibriSpeech --vendor nvidia 
'''