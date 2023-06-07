

### 数据集下载地址(global proxy)
数据来源 https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/wav2vec2#quick-start-guide

执行

DATASET_DIR=[PATH]  bash training/benchmarks/wav2vec2/pytorch/scripts/download_data.sh

DATASET_DIR=[PATH]  bash training/benchmarks/wav2vec2/pytorch/scripts/generate_filelists.sh


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(mAP) | 性能（ntokens/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机8卡  | config_A100x1x8 | 3-4天    | 0.605    | 0.605       | 1363049      |

ps：
因大模型运行较久，参考精度通过resume+ckpt方式训练得到，将如下文件中resume = True && no_save = False 即可支持训练过程保持ckpt，任务断掉后继续接着训练。(wav2vec2_Perf/training/benchmarks/wav2vec2/pytorch/config/_base.py)

* resume=False #[bool], default False, if True, read last_checkpoint from ckpt saved path
* ckpt=None   #[str,path], default None, if True, given ckpt path
* no_save=True #[bool], default True, if True , do not save ckpt